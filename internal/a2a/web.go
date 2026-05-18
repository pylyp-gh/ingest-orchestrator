// Package a2a — web.go serves a minimal single-page UI for testing the
// orchestrator's A2A endpoint interactively. Mounted at GET /ui.
//
// The page is a self-contained HTML document з embedded CSS + JS — no
// build step, no asset directory, no static-files Service. The browser
// uses the same hostname as the agent (ingest.ash.ph.lab), so same-origin
// POSTs to /messages just work without CORS.
//
// Renders a one-textarea form for the user's request, a submit button,
// and a streaming-style response panel. JS does a single sync POST to
// /messages and pretty-prints the resulting Task JSON.
//
// Why not /messages:stream (SSE) for the UI: that endpoint requires
// peer to also implement /messages:respond, which is awkward for an
// interactive human in a browser. Sync /messages with policy-based
// elicitation fallback is the appropriate interactive flow.
package a2a

import (
	_ "embed"
	"net/http"
)

//go:embed playground.html
var playgroundHTML []byte

// PlaygroundHandler serves the single-page A2A playground. Plain GET /ui
// returns the HTML; anything else returns Method Not Allowed.
func PlaygroundHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-cache")
	_, _ = w.Write(playgroundHTML)
}
