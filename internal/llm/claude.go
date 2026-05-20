// Package llm wraps the OpenAI-compatible chat completion API for use with
// agentgateway's Anthropic backend. Despite the package name "claude", the
// SDK on the wire is OpenAI — agentgateway translates OpenAI ChatCompletion
// payloads into Anthropic's native /v1/messages format server-side.
//
// Why OpenAI SDK against an Anthropic model:
//   - agentgateway parses request bodies as OpenAI ChatCompletionRequest
//   - Anthropic's tool_choice shape doesn't deserialise into OpenAI's Rust
//     enum → gateway returns 503 if you send native Anthropic body
//   - OpenAI SDK generates a compatible body → gateway translates outward
//
// Same pattern as kagent's claude-via-gateway ModelConfig — single source
// of truth for the gateway path and translation rules.
package llm

import (
	"context"
	"encoding/json"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pylyp-gh/ingest-orchestrator/internal/tracing"
	"go.opentelemetry.io/otel/attribute"
)

type Claude struct {
	client openai.Client
	model  string
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// New returns a Claude wrapper pointed at agentgateway-proxy. The real
// Anthropic key is held server-side by the AgentgatewayBackend's
// secretRef; we send any non-empty placeholder key so the SDK's auth
// header is populated.
func New() *Claude {
	baseURL := envOr("OPENAI_BASE_URL", "https://agentgateway-proxy.agentgateway-system.svc.cluster.local/v1/")
	apiKey := envOr("OPENAI_API_KEY", "stub-gateway-injects-real-key")
	model := envOr("MODEL_NAME", "claude-opus-4-6")
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	)
	return &Claude{client: client, model: model}
}

// Complete sends a chat-completion request with the given messages and
// (optional) tools. Returns the assistant message — which may be plain
// text or contain tool_calls the caller should execute and feed back.
func (c *Claude) Complete(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolParam,
) (*openai.ChatCompletionMessage, error) {
	ctx, span := tracing.Tracer().Start(ctx, "llm.chat.completions")
	tracing.SetKind(span, tracing.KindLLM)
	span.SetAttributes(
		attribute.String("llm.system", "openai"),
		attribute.String("llm.provider", "anthropic"), // real upstream via gateway
		attribute.String("llm.model_name", c.model),
		attribute.Int("llm.tools.count", len(tools)),
		attribute.Int("llm.messages.count", len(messages)),
	)
	if msgsJSON, jerr := json.Marshal(messages); jerr == nil {
		// Cap aggressively — full message history з system prompt can
		// be 5-10KB, eats span budget and clutters Phoenix UI.
		s := string(msgsJSON)
		if len(s) > 2000 {
			s = s[:2000] + "…"
		}
		span.SetAttributes(attribute.String("input.value", s))
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(c.model),
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	resp, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		tracing.EndWithErr(span, err)
		return nil, err
	}
	if len(resp.Choices) == 0 {
		tracing.EndWithErr(span, ErrNoChoices)
		return nil, ErrNoChoices
	}

	// OpenInference llm.token_count.* — Phoenix cost panel uses these.
	span.SetAttributes(
		attribute.Int64("llm.token_count.prompt", resp.Usage.PromptTokens),
		attribute.Int64("llm.token_count.completion", resp.Usage.CompletionTokens),
		attribute.Int64("llm.token_count.total", resp.Usage.TotalTokens),
		attribute.Int("llm.choices.count", len(resp.Choices)),
		attribute.Int("llm.tool_calls.count", len(resp.Choices[0].Message.ToolCalls)),
	)
	out := resp.Choices[0].Message.Content
	if out == "" && len(resp.Choices[0].Message.ToolCalls) > 0 {
		out = "(tool_calls; no plain text)"
	}
	if len(out) > 480 {
		out = out[:480] + "…"
	}
	span.SetAttributes(attribute.String("output.value", out))
	span.End()
	return &resp.Choices[0].Message, nil
}

// Model returns the configured model name (used in logs).
func (c *Claude) Model() string { return c.model }
