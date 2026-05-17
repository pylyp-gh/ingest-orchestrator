// Package a2a — streaming.go implements POST /messages:stream, an SSE
// endpoint that drives the same agent loop as /messages but emits
// TaskStatus events as the work progresses. Critically, it attaches the
// SSE stream to the request context so the ElicitationHandler can push
// "input-required" events at the peer and block on the peer's reply.
//
// Wire shape (events):
//
//	data: {"task_id":"...","status":"WORKING","timestamp":"..."}
//	data: {"task_id":"...","status":"input-required","correlation_id":"...","message":"...","schema":{...}}
//	data: {"task_id":"...","status":"WORKING","timestamp":"..."}
//	data: {"task_id":"...","status":"COMPLETED","result":{"role":"agent","content":[{"type":"text","text":"..."}]}}
//
// Peer responds to input-required by POSTing to /messages:respond with
// the correlation_id (see respond.go).
package a2a

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
	"time"

	"github.com/google/uuid"
	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
)

// StreamingHandler bundles deps needed by /messages:stream. Same Claude
// + MCP dependencies as the sync handler, plus the PendingRegistry shared
// with /messages:respond so elicit replies route back to the waiting
// handler goroutine.
//
// Mu serialises streaming requests so the registry's single active-stream
// slot is never contested. Concurrent streams would need per-session MCP
// clients — much heavier change, deferred until needed.
type StreamingHandler struct {
	*Handler
	Pending *elicit.PendingRegistry
	Mu      sync.Mutex
}

type streamEvent struct {
	TaskID        string         `json:"task_id"`
	Status        string         `json:"status"`
	Timestamp     time.Time      `json:"timestamp"`
	Message       string         `json:"message,omitempty"`
	CorrelationID string         `json:"correlation_id,omitempty"`
	Schema        any            `json:"schema,omitempty"`
	Result        map[string]any `json:"result,omitempty"`
	Error         string         `json:"error,omitempty"`
}

// SendStreamingMessage handles POST /messages:stream. The connection stays
// open for the duration of the agent loop; events are pushed as the loop
// progresses through tool calls and elicit prompts.
func (h *StreamingHandler) SendStreamingMessage(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req SendMessageRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decode: %v", err), http.StatusBadRequest)
		return
	}
	userText := ""
	for _, c := range req.Message.Content {
		if c.Type == "text" {
			userText += c.Text
		}
	}

	// Serialise streaming requests — registry's active-stream slot is a
	// single-occupant resource. Brief blocking under concurrent load is
	// fine for the lab use case (interactive, one user at a time).
	h.Mu.Lock()
	defer h.Mu.Unlock()

	stream, err := elicit.NewStream(w)
	if err != nil {
		// Should not happen with stock net/http, but surface for debug.
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer stream.Close()

	// Register stream in the process-wide registry so ElicitationHandler
	// (which runs in MCP transport goroutine without our ctx) can push
	// prompts to it. Defer the clear so policy fallback kicks in for any
	// subsequent non-streaming request.
	h.Pending.SetActiveStream(stream)
	defer h.Pending.SetActiveStream(nil)

	taskID := uuid.NewString()
	emit := func(ev streamEvent) {
		ev.TaskID = taskID
		if ev.Timestamp.IsZero() {
			ev.Timestamp = time.Now().UTC()
		}
		if err := stream.SendEvent(ev); err != nil {
			log.Printf("[stream %s] send: %v", taskID, err)
		}
	}

	emit(streamEvent{Status: "WORKING"})

	answer, err := Loop(r.Context(), h.Claude, h.MCP, userText)
	if err != nil {
		emit(streamEvent{Status: "FAILED", Error: err.Error()})
		return
	}
	emit(streamEvent{Status: "COMPLETED", Result: map[string]any{
		"role": "agent",
		"content": []map[string]any{
			{"type": "text", "text": answer},
		},
	}})
}
