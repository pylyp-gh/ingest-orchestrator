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
	"sync"
	"sync/atomic"

	"github.com/google/uuid"
)

// Result is the peer's reply to an elicit request — mirrors mcp.ElicitResult
// at the protocol boundary, but kept in our own type to avoid leaking the SDK
// into HTTP layer code.
type Result struct {
	Action  string         `json:"action"`            // "accept" | "decline" | "cancel"
	Content map[string]any `json:"content,omitempty"` // matches schema when accept
}

// PendingRegistry tracks in-flight elicitation requests waiting on peer
// responses, plus the currently-active SSE stream (if any) that the
// ElicitationHandler should push prompts at.
//
// Why a registry holds the stream instead of plumbing through context:
// MCP SDK invokes ElicitationHandler from an async JSON-RPC dispatcher
// goroutine — its ctx is transport-level, NOT the ctx the caller passed
// to CallTool. context.WithValue plumbing fundamentally cannot reach
// across that boundary. The registry sidesteps that by stashing the
// stream as process-wide state, with serialised access via the caller's
// mutex.
//
// Trade-off: with a single shared stream slot, only ONE streaming
// request can be in flight at a time. The streaming handler enforces
// that via its own mutex. Concurrent streaming would need per-session
// MCP clients — a much heavier change deferred until needed.
type PendingRegistry struct {
	m            sync.Map           // map[string]chan Result
	activeStream atomic.Pointer[any] // *Stream — type erased to avoid import cycle
}

func NewPendingRegistry() *PendingRegistry {
	return &PendingRegistry{}
}

// SetActiveStream registers s as the stream the ElicitationHandler should
// push prompts at. Pass nil to clear (caller should always pair Set with
// a deferred Set(nil) to release). Returns the previous stream so callers
// can chain / restore if needed.
//
// `any` typed to avoid an import loop — the stream type is in the same
// package, callers pass *Stream which type-asserts safely in ActiveStream.
func (p *PendingRegistry) SetActiveStream(s *Stream) {
	if s == nil {
		p.activeStream.Store(nil)
		return
	}
	var v any = s
	p.activeStream.Store(&v)
}

// ActiveStream returns the registered stream, or (nil, false) if none.
func (p *PendingRegistry) ActiveStream() (*Stream, bool) {
	v := p.activeStream.Load()
	if v == nil {
		return nil, false
	}
	s, ok := (*v).(*Stream)
	return s, ok && s != nil
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

