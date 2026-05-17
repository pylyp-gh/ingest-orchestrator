// Package elicit implements the bridge between MCP server-side
// elicitation/create requests and the A2A streaming endpoint of the
// orchestrator.
//
// Architecture:
//
//   - MCP server (doc-writer-mcp) calls ss.Elicit during tool execution.
//     This invokes the ElicitationHandler registered on our MCP client,
//     which by SDK contract blocks until it returns a result.
//
//   - The handler generates a fresh correlation ID, registers a wait
//     channel in PendingMap, and pushes an "input-required" event onto
//     the SSE stream currently servicing the upstream A2A request.
//     Channel-blocking until peer responds.
//
//   - The /messages:respond endpoint deserialises the peer reply, looks
//     up the channel by correlation ID, and sends the result. Handler
//     unblocks, returns to MCP server, server resumes its tool flow.
//
// Concurrency: PendingMap is sync.Map for simplicity; per-request access
// is one-shot (register, wait, delete) so contention is minimal. Channels
// are unbuffered with non-blocking sends from the respond handler — if
// the wait has timed out / cancelled, the send is dropped, not panicked.
package elicit

import (
	"context"
	"sync"

	"github.com/google/uuid"
)

// SSEStreamKey is the context key under which a *Stream is stored by the
// streaming handler before it invokes work that may trigger Elicit. Reader
// is the ElicitationHandler.
type ctxKey int

const SSEStreamKey ctxKey = 1

// Result is the peer's reply to an elicit request — mirrors mcp.ElicitResult
// at the protocol boundary, but kept in our own type to avoid leaking the SDK
// into HTTP layer code.
type Result struct {
	Action  string         `json:"action"`            // "accept" | "decline" | "cancel"
	Content map[string]any `json:"content,omitempty"` // matches schema when accept
}

// PendingRegistry tracks in-flight elicitation requests waiting on peer
// responses. Single instance per orchestrator process, shared by the
// streaming handler and the respond handler.
type PendingRegistry struct {
	m sync.Map // map[string]chan Result
}

func NewPendingRegistry() *PendingRegistry {
	return &PendingRegistry{}
}

// Register reserves a correlation ID and returns a channel that will
// receive the peer's response (if any) and an unregister function the
// handler must call (via defer) to free the map entry — covers both
// success and timeout paths.
func (p *PendingRegistry) Register() (correlationID string, ch <-chan Result, unregister func()) {
	cid := uuid.NewString()
	c := make(chan Result, 1) // buffered: respond handler shouldn't block
	p.m.Store(cid, c)
	unregister = func() { p.m.Delete(cid) }
	return cid, c, unregister
}

// Deliver looks up the channel for correlationID and sends the result
// non-blockingly. Returns true if the result was delivered, false if no
// pending wait exists (already timed out, cancelled, or never registered).
func (p *PendingRegistry) Deliver(correlationID string, r Result) bool {
	v, ok := p.m.Load(correlationID)
	if !ok {
		return false
	}
	c := v.(chan Result)
	select {
	case c <- r:
		return true
	default:
		// buffer full means double-delivery; should not happen with the
		// register-once contract, but tolerate to avoid panic.
		return false
	}
}

// FromContext extracts the active SSE stream from ctx, if any. Returns
// (nil, false) when no stream is attached — the ElicitationHandler then
// falls back to its policy default.
func FromContext(ctx context.Context) (*Stream, bool) {
	s, ok := ctx.Value(SSEStreamKey).(*Stream)
	return s, ok && s != nil
}
