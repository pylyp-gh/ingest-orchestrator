package router

import "context"

// ctxKey is unexported so external packages can't fabricate or read the
// Verdict from a different module's context type. Standard Go idiom.
type ctxKey struct{}

// WithVerdict attaches the classifier's Verdict to the request context.
// Called once at turn start by a2a.Loop; the same Verdict then propagates
// to every llm.Complete invocation inside the tool-use loop without
// needing to thread an extra parameter through Loop / executeToolCall.
func WithVerdict(ctx context.Context, v Verdict) context.Context {
	return context.WithValue(ctx, ctxKey{}, v)
}

// VerdictFromContext returns the Verdict if one is attached, plus a
// boolean ok flag. Callers that don't find one (ROUTER_ENABLED=false,
// classifier failed open, etc.) fall back to their legacy single-model
// path.
func VerdictFromContext(ctx context.Context) (Verdict, bool) {
	v, ok := ctx.Value(ctxKey{}).(Verdict)
	return v, ok
}
