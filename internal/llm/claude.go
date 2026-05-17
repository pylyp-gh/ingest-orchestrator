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
	"os"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
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
	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(c.model),
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}
	resp, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, ErrNoChoices
	}
	return &resp.Choices[0].Message, nil
}

// Model returns the configured model name (used in logs).
func (c *Claude) Model() string { return c.model }
