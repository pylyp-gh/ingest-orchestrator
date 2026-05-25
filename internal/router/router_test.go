package router

import (
	"context"
	"testing"
)

func TestParseTier(t *testing.T) {
	cases := []struct {
		in    string
		want  Tier
		isErr bool
	}{
		{"TIER0", Tier0, false},
		{"tier1", Tier1, false},
		{"TIER2", Tier2, false},
		{"tier3", Tier3, false},
		{"TIER4", Tier2, true},  // unknown defaults to tier2
		{"garbage", Tier2, true},
		{"", Tier2, true},
	}
	for _, c := range cases {
		got, err := ParseTier(c.in)
		if got != c.want {
			t.Errorf("ParseTier(%q) tier = %v, want %v", c.in, got, c.want)
		}
		if (err != nil) != c.isErr {
			t.Errorf("ParseTier(%q) err = %v, want isErr=%v", c.in, err, c.isErr)
		}
	}
}

func TestParseVerdict(t *testing.T) {
	cases := []struct {
		in       string
		wantTier Tier
		hasErr   bool
	}{
		{"TIER1: factual question", Tier1, false},
		{"TIER3 :  deep reasoning needed", Tier3, false},
		{"TIER0 trivial greeting", Tier0, false},
		{"TIER2: multi-step\nadditional rambling", Tier2, false},
		{"", Tier2, true},
		{"nope, no tier here", Tier2, true},
	}
	for _, c := range cases {
		got, err := ParseVerdict(c.in)
		if got.Tier != c.wantTier {
			t.Errorf("ParseVerdict(%q) tier = %v, want %v", c.in, got.Tier, c.wantTier)
		}
		if (err != nil) != c.hasErr {
			t.Errorf("ParseVerdict(%q) err = %v, want hasErr=%v", c.in, err, c.hasErr)
		}
	}
}

func TestContextRoundtrip(t *testing.T) {
	ctx := context.Background()
	if _, ok := VerdictFromContext(ctx); ok {
		t.Fatal("empty context should not carry verdict")
	}
	v := Verdict{Tier: Tier3, Reason: "архітектурне питання"}
	ctx = WithVerdict(ctx, v)
	got, ok := VerdictFromContext(ctx)
	if !ok {
		t.Fatal("verdict missing from context after WithVerdict")
	}
	if got.Tier != Tier3 || got.Reason != "архітектурне питання" {
		t.Errorf("verdict round-trip mismatch: got %+v", got)
	}
}

func TestResolveMissing(t *testing.T) {
	t.Setenv("ROUTER_TIER1_BASE_URL", "")
	t.Setenv("ROUTER_TIER1_MODEL", "")
	if _, _, err := Resolve(Tier1); err == nil {
		t.Fatal("Resolve should error when env vars unset")
	}
}

func TestResolveOk(t *testing.T) {
	t.Setenv("ROUTER_TIER1_BASE_URL", "https://example/v1/haiku/")
	t.Setenv("ROUTER_TIER1_MODEL", "claude-haiku-4-5")
	base, model, err := Resolve(Tier1)
	if err != nil {
		t.Fatalf("Resolve failed: %v", err)
	}
	if base != "https://example/v1/haiku/" || model != "claude-haiku-4-5" {
		t.Errorf("Resolve returned base=%q model=%q", base, model)
	}
}

func TestEstimateCost(t *testing.T) {
	if c := EstimateCost(Tier0, 1_000_000, 1_000_000); c != 0 {
		t.Errorf("Tier0 cost should be 0, got %v", c)
	}
	// Tier1: 1M input @ 0.80 + 1M output @ 4.00 = 4.80
	if c := EstimateCost(Tier1, 1_000_000, 1_000_000); c < 4.79 || c > 4.81 {
		t.Errorf("Tier1 cost 1M+1M should be ~4.80, got %v", c)
	}
}
