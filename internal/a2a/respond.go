// Package a2a — respond.go implements POST /messages:respond, the
// companion endpoint that delivers an A2A peer's reply to a pending
// elicit. The streaming handler put a correlation_id into the SSE event;
// peer echoes it back here along with the actual {action, content}
// payload that satisfies the schema the server posted.
//
// Why a separate endpoint instead of bidirectional streaming:
//   - A2A spec keeps requests and responses as separate POSTs, both with
//     correlation IDs. Mirrors the bidirectional MCP transport pattern.
//   - HTTP request bodies are unidirectional in standard library net/http.
//     Bidi would force WebSocket or HTTP/2 PUSH, both of which have less
//     tooling support (curl can't replay WS, hard to test by hand).
//
// Idempotency: delivering the same correlation_id twice is a no-op (the
// channel buffer holds one slot; second send is dropped). This matches
// the at-most-once semantics of the elicit protocol.
package a2a

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
)

type RespondRequest struct {
	CorrelationID string         `json:"correlation_id"`
	Action        string         `json:"action"`            // "accept" | "decline" | "cancel"
	Content       map[string]any `json:"content,omitempty"` // when action="accept"
}

type RespondHandler struct {
	Pending *elicit.PendingRegistry
}

// Respond handles POST /messages:respond. Body shape:
//
//	{"correlation_id":"...", "action":"accept", "content":{"choice":"new_version"}}
//
// Returns 202 Accepted on successful delivery, 404 if the correlation_id
// is unknown (already timed out or never existed).
func (h *RespondHandler) Respond(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	var req RespondRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("decode: %v", err), http.StatusBadRequest)
		return
	}
	if req.CorrelationID == "" {
		http.Error(w, "correlation_id required", http.StatusBadRequest)
		return
	}
	if req.Action == "" {
		http.Error(w, "action required (accept|decline|cancel)", http.StatusBadRequest)
		return
	}

	ok := h.Pending.Deliver(req.CorrelationID, elicit.Result{
		Action:  req.Action,
		Content: req.Content,
	})
	if !ok {
		log.Printf("[respond] unknown correlation_id=%s (timed out or unseen)", req.CorrelationID)
		http.Error(w, "no pending elicit for that correlation_id", http.StatusNotFound)
		return
	}
	log.Printf("[respond] delivered action=%s for cid=%s", req.Action, req.CorrelationID)
	w.WriteHeader(http.StatusAccepted)
}
