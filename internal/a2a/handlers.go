// Package a2a — handlers.go: HTTP handlers for the A2A protocol endpoints.
//
// SendMessage now drives the Claude tool-use loop (agent.go), feeding the
// user's text through the LLM with the MCP tool list, looping on tool_calls
// until a terminal text response. Replaces the Phase 1 echo behaviour.
package a2a

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
	"github.com/pylyp-gh/ingest-orchestrator/internal/peer"
	"github.com/pylyp-gh/ingest-orchestrator/internal/router"
)

// SendMessageRequest mirrors the A2A spec request shape.
type SendMessageRequest struct {
	Message Message        `json:"message"`
	Config  *Configuration `json:"configuration,omitempty"`
}

type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
}

type Configuration struct {
	ReturnImmediately bool `json:"returnImmediately,omitempty"`
}

type Task struct {
	ID        string    `json:"id"`
	ContextID string    `json:"contextId,omitempty"`
	Status    Status    `json:"status"`
	History   []Message `json:"history,omitempty"`
}

type Status struct {
	State     string    `json:"state"`
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message,omitempty"`
}

// Handler bundles dependencies needed by the A2A handlers: Claude wrapper,
// MCP client session, peer discovery, and the optional Haiku-based tier
// Classifier. Constructed once at startup and reused across requests
// (session reuse → connection keep-alive on both sides). A nil
// Classifier signals that router is disabled; Loop honours that and
// skips the per-turn classification call.
type Handler struct {
	Claude     *llm.Claude
	MCP        *mcpclient.Client
	Discovery  *peer.Discovery
	Classifier *router.Classifier
}

// SendMessage is the A2A POST /messages handler. Drives the Claude agent
// loop and returns the resulting Task with full conversation history.
func (h *Handler) SendMessage(w http.ResponseWriter, r *http.Request) {
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

	task := Task{
		ID:     uuid.NewString(),
		Status: Status{State: "WORKING", Timestamp: time.Now().UTC()},
		History: []Message{req.Message},
	}

	answer, err := Loop(r.Context(), h.Claude, h.MCP, h.Discovery, h.Classifier, userText)
	if err != nil {
		task.Status = Status{
			State:     "FAILED",
			Timestamp: time.Now().UTC(),
			Message:   err.Error(),
		}
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusInternalServerError)
		json.NewEncoder(w).Encode(task)
		return
	}

	task.Status = Status{State: "COMPLETED", Timestamp: time.Now().UTC()}
	task.History = append(task.History, Message{
		Role:    "agent",
		Content: []Content{{Type: "text", Text: answer}},
	})

	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(task)
}
