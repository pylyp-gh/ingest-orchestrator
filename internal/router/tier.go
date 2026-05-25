// Package router implements the LLM-based semantic tier classifier: a
// small Haiku call decides which tier (Ollama, Haiku, Sonnet, Opus) a
// given user request should land on, and the orchestrator's tool-use
// loop honours that choice for every llm.Complete inside the turn.
//
// Layered split:
//   tier.go      — Tier enum, env-driven resolution to (baseURL, model)
//   verdict.go   — Verdict struct, parser for the classifier's reply
//   context.go   — context.Context helpers for handing the Verdict to the
//                  llm.Complete deep in the call stack
//   pricing.go   — per-tier $/1M token table, used only for span attrs
//   classifier.go — the Haiku-backed classifier itself
//
// Turn flow:
//   1. cmd/orchestrator main wires a *Classifier on startup if
//      ROUTER_ENABLED=true; passes it to the a2a Handler.
//   2. a2a.Loop on every new user turn calls classifier.Classify once,
//      attaches the Verdict to the request context via WithVerdict.
//   3. llm.Complete reads VerdictFromContext and, if present, swaps the
//      OpenAI base URL + model for the tier's values before issuing the
//      chat completion.
//   4. All spans inside the loop (orchestrator.loop, llm.chat.completions,
//      tool.call.*) carry router.tier + router.reason for post-hoc
//      cost/quality analytics in Phoenix.
package router

import (
	"fmt"
	"os"
)

// Tier identifies one of the four cost/capability tiers. The numeric
// value is intentional, not just an iota: Tier0 < Tier1 < Tier2 < Tier3
// in cost and (loosely) capability, so we can compare with < and bump
// up via escalation rules without lookup tables.
type Tier int

const (
	Tier0 Tier = iota // Ollama qwen2.5:7b — local, free
	Tier1             // Claude Haiku 4.5 — cheap cloud
	Tier2             // Claude Sonnet 4.6 — standard
	Tier3             // Claude Opus 4.7 — premium
)

// String returns the canonical lowercase tier name used in env-var keys
// and Phoenix span attributes (e.g. router.tier="tier2").
func (t Tier) String() string {
	switch t {
	case Tier0:
		return "tier0"
	case Tier1:
		return "tier1"
	case Tier2:
		return "tier2"
	case Tier3:
		return "tier3"
	default:
		return fmt.Sprintf("tier?%d", int(t))
	}
}

// ParseTier converts the classifier's verdict token ("TIER0".."TIER3")
// into a Tier. Returns Tier2 (the safe middle ground) and a non-nil
// error if the input is malformed: caller can fall back without aborting.
func ParseTier(s string) (Tier, error) {
	switch s {
	case "TIER0", "tier0":
		return Tier0, nil
	case "TIER1", "tier1":
		return Tier1, nil
	case "TIER2", "tier2":
		return Tier2, nil
	case "TIER3", "tier3":
		return Tier3, nil
	default:
		return Tier2, fmt.Errorf("router: unknown tier token %q (defaulting to tier2)", s)
	}
}

// Resolve looks up the OpenAI base URL and model name for a Tier from
// the process environment. Keys are
//   ROUTER_TIER<N>_BASE_URL, ROUTER_TIER<N>_MODEL.
// Returns ("", "", error) if either is unset, signalling that the caller
// should not enable the router and fall back to legacy
// OPENAI_BASE_URL / MODEL_NAME.
func Resolve(t Tier) (baseURL, model string, err error) {
	keyBase := fmt.Sprintf("ROUTER_%s_BASE_URL", upper(t.String()))
	keyModel := fmt.Sprintf("ROUTER_%s_MODEL", upper(t.String()))
	baseURL = os.Getenv(keyBase)
	model = os.Getenv(keyModel)
	if baseURL == "" || model == "" {
		return "", "", fmt.Errorf("router: %s or %s unset", keyBase, keyModel)
	}
	return baseURL, model, nil
}

func upper(s string) string {
	out := make([]byte, len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c >= 'a' && c <= 'z' {
			c -= 'a' - 'A'
		}
		out[i] = c
	}
	return string(out)
}
