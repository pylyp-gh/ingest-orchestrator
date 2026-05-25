package router

import (
	"context"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/openai/openai-go"
	"github.com/openai/openai-go/option"
	"github.com/pylyp-gh/ingest-orchestrator/internal/tracing"
	"go.opentelemetry.io/otel/attribute"
)

// classifierSystemPrompt — Ukrainian per project convention, but the
// verdict line itself uses ASCII tokens so ParseVerdict has zero
// locale ambiguity. The "anti-patterns" coda mirrors router-agent's
// kagent system prompt, keeping the two semantically aligned even
// though they run in different processes.
const classifierSystemPrompt = `Ти tier router. Класифікуй кожен user request у один з чотирьох tier-ів і поверни ВИКЛЮЧНО один рядок такого формату:

TIER<N>: <одне речення reason українською>

де <N> = 0, 1, 2 або 3. Більше нічого: ніяких преамбул, ніяких post-script ліній, ніякого Markdown.

Критерії tier-ів:

TIER0 (Ollama qwen2.5:7b, локальна, безкоштовна):
- Привітання, формальні відповіді ("привіт", "дякую", "як справи").
- Trivial lookups з фіксованим coverage (поточний час, простий обчислення).
- Запит коротший за 20 символів без технічного контексту.

TIER1 (Claude Haiku 4.5, швидка cloud):
- Single-shot factual Q&A ("що таке Kubernetes Pod?", "як працює TCP?").
- Definitional / concept-level питання без tool-use.
- Запит на стислі пояснення.

TIER2 (Claude Sonnet 4.6, multi-domain з tools):
- Multi-step tasks що потребують tool calls (MCP add_document, peer delegations).
- Cluster operations ("переглянь pods", "знайди errors у logs").
- Code reading / debugging.
- Coordination across domains (k8s + helm + observability + docs ingest).

TIER3 (Claude Opus 4.7, глибокий reasoning):
- Architecture / design decisions ("ArgoCD vs Flux для multi-cluster?").
- Security review / threat modeling.
- Comparative analysis з trade-offs.
- Запит довший за 2000 символів (escalate один tier вище незалежно від теми).

Anti-patterns:
- НЕ додавай інші слова до TIER token.
- НЕ пиши Markdown (без ###, **, code blocks).
- НЕ виправдовуй вибір довше ніж одне речення.`

// Classifier wraps a dedicated OpenAI client pinned to the Haiku tier.
// Kept separate from the main llm.Claude wrapper so the classifier
// stays cheap and predictable: it never receives the user-loop tools,
// never escalates, and never inherits the user-turn fallback chain.
type Classifier struct {
	client openai.Client
	model  string
	label  string
}

// NewClassifier wires the classifier from ROUTER_CLASSIFIER_BASE_URL +
// ROUTER_CLASSIFIER_MODEL. Returns (nil, nil) if either env var is
// unset (router disabled, caller continues without classification).
// Returns an error only on hard misconfigurations.
func NewClassifier() (*Classifier, error) {
	baseURL := os.Getenv("ROUTER_CLASSIFIER_BASE_URL")
	model := os.Getenv("ROUTER_CLASSIFIER_MODEL")
	if baseURL == "" || model == "" {
		return nil, nil
	}
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = "stub-gateway-injects-real-key"
	}
	client := openai.NewClient(
		option.WithBaseURL(baseURL),
		option.WithAPIKey(apiKey),
	)
	c := &Classifier{
		client: client,
		model:  model,
		label:  "haiku-classifier",
	}
	log.Printf("router: classifier configured: base=%s model=%s", baseURL, model)
	return c, nil
}

// Classify issues a single chat completion to the Haiku tier asking it
// to label the user prompt. Emits a Phoenix span "router.classify"
// carrying the prompt, the verdict, token counts, and estimated cost
// so cost-of-routing is visible alongside cost-of-doing-the-task.
//
// Never returns an error to the caller in practice: if the classifier
// HTTP call fails or the reply is unparsable, we return (Tier2, "reason
// reflecting fallback") so the loop continues without aborting. The
// returned error is informational; log it but don't fail the turn.
func (c *Classifier) Classify(ctx context.Context, userText string) (Verdict, error) {
	ctx, span := tracing.Tracer().Start(ctx, "router.classify")
	tracing.SetKind(span, tracing.KindLLM)
	span.SetAttributes(
		attribute.String("llm.system", "openai"),
		attribute.String("llm.provider", "anthropic"),
		attribute.String("llm.model_name", c.model),
		attribute.String("input.value", tracing.SafeTrunc(userText, 1000)),
	)
	defer span.End()

	msgs := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(classifierSystemPrompt),
		openai.UserMessage(userText),
	}
	params := openai.ChatCompletionNewParams{
		Model:     openai.ChatModel(c.model),
		Messages:  msgs,
		MaxTokens: openai.Int(64),
	}

	resp, err := c.client.Chat.Completions.New(ctx, params)
	if err != nil {
		span.SetAttributes(attribute.String("output.value",
			fmt.Sprintf("classifier error, defaulting to tier2: %v", err)))
		tracing.EndWithErr(span, err)
		return Verdict{Tier: Tier2, Reason: "classifier HTTP error"}, err
	}
	if len(resp.Choices) == 0 {
		err = fmt.Errorf("router: classifier returned zero choices")
		span.SetAttributes(attribute.String("output.value", err.Error()))
		tracing.EndWithErr(span, err)
		return Verdict{Tier: Tier2, Reason: "classifier returned zero choices"}, err
	}

	raw := strings.TrimSpace(resp.Choices[0].Message.Content)
	verdict, perr := ParseVerdict(raw)

	span.SetAttributes(
		attribute.String("output.value", raw),
		attribute.String("router.tier", verdict.Tier.String()),
		attribute.String("router.reason", verdict.Reason),
		attribute.Int64("llm.token_count.prompt", resp.Usage.PromptTokens),
		attribute.Int64("llm.token_count.completion", resp.Usage.CompletionTokens),
		attribute.Int64("llm.token_count.total", resp.Usage.TotalTokens),
		attribute.Float64("llm.cost.estimated",
			EstimateCost(Tier1, resp.Usage.PromptTokens, resp.Usage.CompletionTokens)),
	)

	if perr != nil {
		log.Printf("router: verdict parse warning: %v (raw=%q)", perr, raw)
	}
	return verdict, perr
}
