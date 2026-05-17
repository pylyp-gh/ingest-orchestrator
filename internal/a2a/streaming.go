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
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
)

// StreamingHandler bundles deps needed by /messages:stream. Same Claude
// + MCP dependencies as the sync handler, plus the PendingRegistry shared
// with /messages:respond so elicit replies route back to the waiting
// handler goroutine.
type StreamingHandler struct {
	*Handler
	Pending *elicit.PendingRegistry
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

	stream, err := elicit.NewStream(w)
	if err != nil {
		// Should not happen with stock net/http, but surface for debug.
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	defer stream.Close()

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

	// Attach SSE stream to context so ElicitationHandler can push events at
	// the peer when invoked deep inside CallTool. The PendingRegistry is
	// closed over by the handler at construction time, no ctx plumbing
	// needed for it.
	ctx := context.WithValue(r.Context(), elicit.SSEStreamKey, stream)

	answer, err := Loop(ctx, h.Claude, h.MCP, userText)
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
