// ingest-orchestrator — A2A agent that demonstrates a full-capability MCP
// client integrated with Anthropic Claude. Acts as a bridge between external
// A2A peers and the doc-writer-mcp MCP server, surfacing Elicitation and
// Sampling callbacks back to the calling peer.
//
// Phase 1 (this file): HTTP server with Agent Card + echo skill — minimum
// viable A2A endpoint for Lab 4 #2 deliverable. Subsequent phases add MCP
// client integration (doc-writer-mcp.add_document), Sampling bridge to
// Anthropic, and Elicitation bridge through A2A streaming.
package main

import (
	"context"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/pylyp-gh/ingest-orchestrator/internal/a2a"
	"github.com/pylyp-gh/ingest-orchestrator/internal/agentcard"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
)

var (
	httpAddr = flag.String("http", ":8080", "HTTP listen address")
)

func main() {
	flag.Parse()
	if err := run(); err != nil {
		fmt.Fprintf(os.Stderr, "fatal: %v\n", err)
		os.Exit(1)
	}
}

func run() error {
	mux := http.NewServeMux()

	// Initialise dependencies — Claude wrapper (OpenAI SDK pointed at the
	// gateway) and MCP client (Streamable HTTP to doc-writer-mcp). Both
	// constructed once and reused across requests for connection keep-alive.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	claude := llm.New()
	log.Printf("claude wrapper configured: model=%s", claude.Model())

	mc, err := mcpclient.New(ctx)
	if err != nil {
		return fmt.Errorf("init mcp client: %w", err)
	}
	defer mc.Close()
	log.Printf("mcp client connected: %d tools available", len(mc.Tools()))

	handler := &a2a.Handler{Claude: claude, MCP: mc}

	// A2A discovery — the Agent Card. Per RFC 8615 well-known URIs, this is
	// the canonical entry point for peers to learn about the agent.
	mux.HandleFunc("/.well-known/agent-card.json", agentcard.Handler())

	// A2A SendMessage — drives the Claude tool-use loop calling MCP tools.
	mux.HandleFunc("/messages", handler.SendMessage)

	// Liveness probe — used by K8s readiness/liveness probes if deployed.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              *httpAddr,
		Handler:           logMiddleware(mux),
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      120 * time.Second, // long enough for LLM round-trips
		IdleTimeout:       60 * time.Second,
	}

	// Graceful shutdown — handle SIGTERM/SIGINT so K8s pod termination
	// drains in-flight requests instead of dropping them. 10s grace period
	// matches typical pod terminationGracePeriodSeconds.
	stopCh := make(chan os.Signal, 1)
	signal.Notify(stopCh, syscall.SIGINT, syscall.SIGTERM)
	errCh := make(chan error, 1)
	go func() {
		log.Printf("ingest-orchestrator listening on %s", *httpAddr)
		if err := srv.ListenAndServe(); err != nil && !errors.Is(err, http.ErrServerClosed) {
			errCh <- err
		}
		close(errCh)
	}()
	select {
	case err := <-errCh:
		return err
	case <-stopCh:
		log.Printf("shutdown signal received, draining...")
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		return srv.Shutdown(ctx)
	}
}

// logMiddleware — minimal request log. Future: replace with structured
// logging (slog) and add request ID propagation for tracing.
func logMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		log.Printf("%s %s %s %v", r.RemoteAddr, r.Method, r.URL.Path, time.Since(start))
	})
}
