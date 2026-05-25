package router

// Pricing is the per-million-token cost of a tier. Used only to compute
// the llm.cost.estimated span attribute. Source: Anthropic public
// pricing as of 2026-05. Update when prices shift; an extra 10% drift
// won't materially change tier-distribution analytics in Phoenix.
type Pricing struct {
	InputPer1M  float64
	OutputPer1M float64
}

// pricingByTier — keep keys aligned with the Tier enum order. Tier 0
// (Ollama local) is free.
var pricingByTier = map[Tier]Pricing{
	Tier0: {InputPer1M: 0, OutputPer1M: 0},
	Tier1: {InputPer1M: 0.80, OutputPer1M: 4.00},  // Haiku 4.5
	Tier2: {InputPer1M: 3.00, OutputPer1M: 15.00}, // Sonnet 4.6
	Tier3: {InputPer1M: 15.0, OutputPer1M: 75.00}, // Opus 4.7
}

// PriceOf returns the configured pricing for a tier. Unknown tier falls
// back to Tier2 (sane middle ground) so a malformed verdict never
// crashes the cost calculation path.
func PriceOf(t Tier) Pricing {
	if p, ok := pricingByTier[t]; ok {
		return p
	}
	return pricingByTier[Tier2]
}

// EstimateCost returns the dollar cost of a single Complete call for a
// given tier, given prompt and completion token counts. Used to set the
// llm.cost.estimated span attribute.
func EstimateCost(t Tier, promptTokens, completionTokens int64) float64 {
	p := PriceOf(t)
	return float64(promptTokens)/1_000_000*p.InputPer1M +
		float64(completionTokens)/1_000_000*p.OutputPer1M
}
