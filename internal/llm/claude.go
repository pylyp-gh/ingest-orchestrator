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
//
// TODO(open question): архітектурний gap навколо 4xx-driven fallback.
//
//   Поточний стан:
//     * Client-side retry на 4xx (Complete() falls back to Ollama if
//       OLLAMA_FALLBACK_URL set). Це fix для orchestrator's own LLM
//       calls, але peers (kagent agents: k8s-agent, helm-agent,
//       doc-agent, observability-agent, etc.) не мають analogous
//       fallback — їх ModelConfig is single-provider, ADK code не
//       робить retry. На credit-burn:
//
//         orchestrator.LLM → 400 → Ollama fallback ✓
//             ↓ delegate_to_kagent_peer
//             peer.LLM → 400 → no fallback → peer returns error
//             ↓
//             orchestrator.LLM (Ollama) бачить error, often outputs
//             free-form prose замість structured tool_calls retry.
//
//   Що ми вже спробували і чому не покращує:
//     * agentgateway `AgentgatewayBackend.spec.ai.groups[]` — failover
//       lights up тільки на provider-side infra failure (5xx/timeout),
//       не на AI-level 4xx like credit_balance_too_low. Confirmed via
//       gateway logs — retry attempts hit same group, eventually surface
//       400 to caller.
//     * `HTTPRoute.spec.rules[].retry` з codes=[400] — gateway re-issues
//       to same backend group, не cross-group. Wasted 3x latency, same
//       result.
//
//   Можливі шляхи розв'язання:
//     1. Підняти upstream feature request у agentgateway: declarative
//        4xx-failover codes per backend (groups[].failoverCodes:
//        [400, 402, 429] etc.). Дозволяє єдиний YAML knob для всіх
//        consumers — peers, orchestrator, sample_client — одночасно.
//        Hardest у часі, але правильний шар.
//     2. Switch primary provider до Ollama локально для всіх kagent
//        peers + orchestrator. ModelConfig change в kagent ns
//        (provider.openai з host=ollama.svc.cluster.local). Втрачаємо
//        Claude tool-use quality (~15-20%), але стек з нульовою cost
//        dependency. Pragmatic для lab demos.
//     3. Sidecar pattern — small Go shim sitting між кожним agent's pod
//        і agentgateway, що catches 4xx і retries against alternate
//        backend. Same logic як наш Complete() але externalised. Heavy.
//     4. Wait for Anthropic credit auto-replenishment / Plan upgrade.
//        Не fix, just delays the problem.
//
//   Поки що: orchestrator survives credit-burn (degraded mode), peers
//   ні. Це **known limitation** документоване у trace tree як missing
//   spans (peer delegation returns error, не traced subtree).
package llm

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
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

// extractMessageText pulls the plain-text content out of an OpenAI
// SDK ChatCompletionMessageParamUnion. Used by the prompt-caching path
// to wrap the system message text у an Anthropic-shape cache_control
// block. Only handles string content (system message у our case);
// returns empty string for other shapes.
func extractMessageText(msg openai.ChatCompletionMessageParamUnion) string {
	if msg.OfSystem != nil {
		if msg.OfSystem.Content.OfString.Valid() {
			return msg.OfSystem.Content.OfString.Value
		}
	}
	return ""
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
		// be 5-10KB, eats span budget and clutters Phoenix UI. Use
		// rune-aware truncation so we don't split a multi-byte UTF-8
		// codepoint and break OTLP gRPC marshaling.
		span.SetAttributes(attribute.String("input.value",
			tracing.SafeTrunc(string(msgsJSON), 2000)))
	}

	params := openai.ChatCompletionNewParams{
		Model:    openai.ChatModel(c.primaryModel),
		Messages: messages,
	}
	if len(tools) > 0 {
		params.Tools = tools
	}

	// Optional prompt caching — opt-in via ENABLE_PROMPT_CACHE=true.
	// Anthropic exposes ephemeral cache for system message + tools що
	// drops cached-input cost ~10x. The OpenAI SDK on our wire doesn't
	// model cache_control natively, so we inject markers via JSON-Set
	// patches on the outgoing request body. agentgateway's
	// OpenAI→Anthropic translator may or may not forward these
	// unmodified — if it strips them, the call still succeeds (just
	// без cache benefit), so це low-risk experiment.
	var callOpts []option.RequestOption
	if os.Getenv("ENABLE_PROMPT_CACHE") == "true" && len(messages) > 0 {
		callOpts = append(callOpts,
			// Anthropic beta gate для prompt caching feature.
			option.WithHeader("anthropic-beta", "prompt-caching-2024-07-31"),
			// Replace messages[0].content (system message) з array
			// of typed blocks з cache_control marker.
			option.WithJSONSet("messages.0.content", []map[string]any{
				{
					"type":          "text",
					"text":          extractMessageText(messages[0]),
					"cache_control": map[string]string{"type": "ephemeral"},
				},
			}),
		)
		// Mark the tools array як cacheable теж — tools schema rarely
		// changes between iterations, so caching saves significant
		// token-pricing on long agent loops.
		if len(tools) > 0 {
			// sjson path для last tool index — caches all tools up to
			// that point. cache_control on last tool у array makes
			// everything before it cacheable per Anthropic semantics.
			lastIdx := fmt.Sprintf("tools.%d.cache_control", len(tools)-1)
			callOpts = append(callOpts,
				option.WithJSONSet(lastIdx, map[string]string{"type": "ephemeral"}),
			)
		}
	}

	resp, err := c.primary.Chat.Completions.New(ctx, params, callOpts...)
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
	// Ollama qwen output can contain CJK characters when fallback fires —
	// rune-aware truncation is mandatory here.
	span.SetAttributes(attribute.String("output.value", tracing.SafeTrunc(out, 480)))
	span.End()
	return &resp.Choices[0].Message, nil
}

// Model returns the configured primary model name (used in logs).
func (c *Claude) Model() string { return c.primaryModel }
