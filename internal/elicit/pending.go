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
//   - For the streaming endpoint, a NEW MCP session is created per
//     /messages:stream request, with an ElicitationHandler closure that
//     captures the stream. Concurrent streams are fully isolated — no
//     shared state across requests, no head-of-line blocking.
//
//   - The PendingRegistry tracks only correlation IDs → response channels.
//     IDs are universally unique, so concurrent streams share the same
//     registry safely; each correlation ID belongs to exactly one stream
//     by construction.
//
//   - The /messages:respond endpoint deserialises the peer reply, looks
//     up the channel by correlation ID, and sends the result. Handler
//     unblocks, returns to MCP server, server resumes its tool flow.
//
// Concurrency: PendingMap is sync.Map for simplicity; per-request access
// is one-shot (register, wait, delete) so contention is minimal. Channels
// are buffered (size 1) so the respond handler never blocks.
package elicit

import (
	"sync"

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
// responses. Correlation IDs are globally unique (uuidv4) so the same
// registry can safely serve any number of concurrent streams.
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
