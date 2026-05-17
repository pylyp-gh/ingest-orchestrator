// Package a2a implements the Agent-to-Agent (A2A) protocol HTTP endpoints
// per a2a-protocol.org specification: SendMessage (POST /messages), GetTask,
// ListTasks, streaming variants, etc.
//
// Phase 1 covers the minimum needed to participate in A2A: SendMessage with
// the "echo" skill. Later phases bridge to MCP server tools and surface
// Sampling/Elicitation callbacks to the calling A2A peer.
package a2a

import (
	"encoding/json"
	"fmt"
	"net/http"
	"time"

	"github.com/google/uuid"
)

// SendMessageRequest mirrors the A2A spec request shape. Subset for Phase 1:
// just the message field is required. Configuration/metadata can be added
// later for streaming/push notifications.
type SendMessageRequest struct {
	Message Message        `json:"message"`
	Config  *Configuration `json:"configuration,omitempty"`
}

type Message struct {
	Role    string    `json:"role"`
	Content []Content `json:"content"`
}

type Content struct {
	Type string `json:"type"` // "text" only in Phase 1
	Text string `json:"text,omitempty"`
}

type Configuration struct {
	ReturnImmediately bool `json:"returnImmediately,omitempty"`
}

// Task mirrors A2A Task object. Phase 1 keeps it simple — no contextId,
// no artifacts; just status + history.
type Task struct {
	ID        string    `json:"id"`
	ContextID string    `json:"contextId,omitempty"`
	Status    Status    `json:"status"`
	History   []Message `json:"history,omitempty"`
}

type Status struct {
	State     string    `json:"state"`     // SUBMITTED | WORKING | COMPLETED | FAILED | etc.
	Timestamp time.Time `json:"timestamp"`
	Message   string    `json:"message,omitempty"`
}

// SendMessageHandler handles POST /messages — the entry point for A2A
// task initiation. For Phase 1, it routes the request through the only
// implemented skill (echo) and returns the result synchronously.
//
// Future: dispatch via skill ID, support streaming via SSE, persist tasks
// in memory/store for later GET /tasks/{id} retrieval.
func SendMessageHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
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
		// Echo skill — Phase 1 only.
		response := echo(userText)
		task := Task{
			ID: uuid.NewString(),
			Status: Status{
				State:     "COMPLETED",
				Timestamp: time.Now().UTC(),
			},
			History: []Message{
				req.Message,
				{Role: "agent", Content: []Content{{Type: "text", Text: response}}},
			},
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(task)
	}
}

// echo — Phase 1 minimum skill. Just bounces text back, useful for
// connectivity validation and protocol shape verification.
func echo(text string) string {
	if text == "" {
		return "(empty input — A2A SendMessage received with no text content)"
	}
	return fmt.Sprintf("echo: %s", text)
}
