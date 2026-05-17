// Package elicit — sse.go implements a minimal Server-Sent Events writer.
//
// SSE framing per WHATWG:
//
//	data: <JSON line>\n
//	\n
//
// Multi-line data uses repeated "data:" lines. We pre-marshal events to a
// single-line JSON string, so the simplest one-line-per-event form works.
//
// Stream lifetime tied to the http.ResponseWriter, which must implement
// http.Flusher (standard for net/http). Each Write+Flush atomically
// guarded by a mutex so multiple goroutines (the handler executing tools,
// the elicit bridge pushing prompts) can share one stream safely.
package elicit

import (
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
)

// Stream is a thread-safe SSE writer attached to a single HTTP response.
type Stream struct {
	w     http.ResponseWriter
	flush http.Flusher
	mu    sync.Mutex
	done  bool
}

// NewStream wraps w as an SSE stream. Sets the required response headers
// and immediately flushes them so the client sees them right away.
// Returns an error if the connection does not support flushing
// (e.g. behind a buffering proxy with no streaming support).
func NewStream(w http.ResponseWriter) (*Stream, error) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return nil, fmt.Errorf("response writer does not support flushing — cannot stream SSE")
	}
	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	// Defensive against proxies that buffer SSE.
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	flusher.Flush()
	return &Stream{w: w, flush: flusher}, nil
}

// SendEvent serialises payload to JSON and writes one SSE frame, flushing
// immediately. Concurrent calls are serialised by an internal mutex.
func (s *Stream) SendEvent(payload any) error {
	if s == nil {
		return fmt.Errorf("send: nil stream")
	}
	b, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("send: marshal: %w", err)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.done {
		return fmt.Errorf("send: stream closed")
	}
	if _, err := fmt.Fprintf(s.w, "data: %s\n\n", b); err != nil {
		return fmt.Errorf("send: write: %w", err)
	}
	s.flush.Flush()
	return nil
}

// Close marks the stream as no longer writable. Idempotent. Does not
// terminate the underlying connection — that happens when the handler
// returns from ServeHTTP.
func (s *Stream) Close() {
	if s == nil {
		return
	}
	s.mu.Lock()
	s.done = true
	s.mu.Unlock()
}
