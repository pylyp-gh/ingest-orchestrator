// Package a2a — streaming.go implements POST /messages:stream, an SSE
// endpoint that drives the same agent loop as /messages but emits
// TaskStatus events as the work progresses.
//
// Each streaming request creates its OWN MCP session with an
// ElicitationHandler closure that captures this request's SSE stream.
// That gives full isolation between concurrent streams — no shared
// active-stream slot, no mutex, no head-of-line blocking. The cost is
// one MCP initialize handshake + ListTools per stream (~100-300ms,
// I/O-bound, fine for interactive use).
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
	"time"

	"github.com/google/uuid"
	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
)

// StreamingHandler bundles deps needed by /messages:stream. The shared
// MCP client from Handler is NOT used — streaming creates per-request
// sessions instead. The shared client serves only the sync /messages
// endpoint, where the ElicitationHandler's policy fallback applies.
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

	// Create per-request MCP session whose ElicitationHandler closure
	// captures THIS stream. Concurrent streams each get their own session,
	// each handler sees only its own stream. No shared mutable state.
	perReq, err := mcpclient.NewWithStream(r.Context(), h.Claude, h.Pending, stream)
	if err != nil {
		emit(streamEvent{Status: "FAILED", Error: fmt.Sprintf("init mcp session: %v", err)})
		return
	}
	defer perReq.Close()

	answer, err := Loop(r.Context(), h.Claude, perReq, h.Discovery, userText)
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
