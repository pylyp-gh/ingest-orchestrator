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
	"errors"
	"log"
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pylyp-gh/ingest-orchestrator/internal/tracing"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"
)

type Claude struct {
	primary       openai.Client
	primaryModel  string
	primaryLabel  string

	// Optional fallback client used when primary returns a 4xx error
	// (credit-burn, auth fail, validation). nil if not configured via env.
	fallback       *openai.Client
	fallbackModel  string
	fallbackLabel  string
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
//
// If OLLAMA_FALLBACK_URL is set, also constructs a secondary client
// pointed at that endpoint. Used by Complete() on 4xx responses from
// the primary — typical scenarios: Anthropic credit-burn (400 with
// credit_balance_too_low body), auth fail (401), rate-limit (429).
// agentgateway's `groups[]` failover only fires on connection-level
// failures (5xx/timeout), not on AI-level error codes, so we handle
// these client-side.
func New() *Claude {
	baseURL := envOr("OPENAI_BASE_URL", "https://agentgateway-proxy.agentgateway-system.svc.cluster.local/v1/")
	apiKey := envOr("OPENAI_API_KEY", "stub-gateway-injects-real-key")
	model := envOr("MODEL_NAME", "claude-opus-4-6")
	primary := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	)

	c := &Claude{
		primary:      primary,
		primaryModel: model,
		primaryLabel: "claude",
	}

	if fbURL := os.Getenv("OLLAMA_FALLBACK_URL"); fbURL != "" {
		fbModel := envOr("OLLAMA_FALLBACK_MODEL", "qwen2.5:14b")
		fb := openai.NewClient(
			option.WithBaseURL(fbURL),
			// Ollama OpenAI-compat endpoint doesn't validate the key.
			option.WithAPIKey("ollama"),
		)
		c.fallback = &fb
		c.fallbackModel = fbModel
		c.fallbackLabel = "ollama-" + fbModel
		log.Printf("llm: fallback configured — url=%s model=%s", fbURL, fbModel)
	}

	return c
}

// isFallbackEligibleErr reports whether the openai-go SDK error is one
// that we expect a different provider to succeed on. 4xx responses
// = AI-level error (credit/auth/quota/validation) — different upstream
// може mati гроші / auth / quota. 5xx-style errors propagate як-is so
// the caller sees them.
func isFallbackEligibleErr(err error) bool {
	var apiErr *openai.Error
	if !errors.As(err, &apiErr) {
		return false
	}
	return apiErr.StatusCode >= 400 && apiErr.StatusCode < 500
}

// Complete sends a chat-completion request with the given messages and
// (optional) tools. Returns the assistant message — which may be plain
// text or contain tool_calls the caller should execute and feed back.
func (c *Claude) Complete(
	ctx context.Context,
	messages []openai.ChatCompletionMessageParamUnion,
	tools []openai.ChatCompletionToolParam,
) (*openai.ChatCompletionMessage, error) {
	// One root LLM span per logical Complete call. If we fall back to a
	// secondary provider mid-flight, we record that as an event +
	// attribute on the same span rather than spawning a sibling span,
	// because semantically це **one** logical model invocation —
	// fallback is a transport-level detail, not a separate operation.
	ctx, span := tracing.Tracer().Start(ctx, "llm.chat.completions")
	tracing.SetKind(span, tracing.KindLLM)
	span.SetAttributes(
		attribute.String("llm.system", "openai"),
		attribute.String("llm.provider", "anthropic"), // real upstream via gateway
		attribute.String("llm.model_name", c.primaryModel),
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
		Model:    openai.ChatModel(c.primaryModel),
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	resp, err := c.primary.Chat.Completions.New(ctx, params)
	usedLabel := c.primaryLabel
	usedModel := c.primaryModel

	if err != nil && c.fallback != nil && isFallbackEligibleErr(err) {
		// Record the primary failure as a span event так Phoenix shows
		// провенанс — first attempt visible навіть якщо ми recover.
		var apiErr *openai.Error
		_ = errors.As(err, &apiErr)
		span.AddEvent("llm.primary.failed", trace.WithAttributes(
			attribute.String("llm.fallback.trigger", apiErr.Message),
			attribute.Int("llm.primary.status_code", apiErr.StatusCode),
		))
		log.Printf("llm: primary %s failed (HTTP %d %s) — falling back to %s",
			c.primaryLabel, apiErr.StatusCode, apiErr.Type, c.fallbackLabel)

		params.Model = openai.ChatModel(c.fallbackModel)
		resp, err = c.fallback.Chat.Completions.New(ctx, params)
		usedLabel = c.fallbackLabel
		usedModel = c.fallbackModel
		span.SetAttributes(
			attribute.Bool("llm.fallback.used", true),
			attribute.String("llm.fallback.provider", c.fallbackLabel),
		)
	}

	span.SetAttributes(
		attribute.String("llm.actual.provider", usedLabel),
		attribute.String("llm.actual.model", usedModel),
	)

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

// Model returns the configured primary model name (used in logs).
func (c *Claude) Model() string { return c.primaryModel }
