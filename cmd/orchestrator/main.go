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
	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
	"github.com/pylyp-gh/ingest-orchestrator/internal/peer"
	"github.com/pylyp-gh/ingest-orchestrator/internal/router"
	"github.com/pylyp-gh/ingest-orchestrator/internal/tracing"
	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/propagation"
)

// extractTraceContext — server-side propagation middleware. Same pattern
// as doc-writer-mcp: лиш extract incoming traceparent, не emit HTTP-level
// span. Keeps agent.loop / claude / tool.call spans як top-level under
// the caller's trace, without polluting Phoenix з HTTP-hop noise.
func extractTraceContext(next http.Handler) http.Handler {
	propagator := otel.GetTextMapPropagator()
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ctx := propagator.Extract(r.Context(), propagation.HeaderCarrier(r.Header))
		next.ServeHTTP(w, r.WithContext(ctx))
	})
}

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

	// OTel tracing — exports OTLP gRPC до Phoenix коли
	// OTEL_EXPORTER_OTLP_ENDPOINT is set, no-op otherwise.
	shutdownTracer, err := tracing.Init(ctx, "ingest-orchestrator", "0.1.0")
	if err != nil {
		log.Printf("WARN: tracing init failed (%v) — continuing без telemetry", err)
		shutdownTracer = func(context.Context) error { return nil }
	}
	defer func() {
		shutdownCtx, c := context.WithTimeout(context.Background(), 5*time.Second)
		defer c()
		_ = shutdownTracer(shutdownCtx)
	}()

	claude := llm.New()
	log.Printf("claude wrapper configured: model=%s", claude.Model())

	// Tier classifier: gated by ROUTER_ENABLED=true. When enabled and the
	// ROUTER_CLASSIFIER_* env vars are set, every new user turn passes
	// through a single Haiku classification call before the main tool-use
	// loop; the chosen tier propagates into llm.Complete via context. When
	// disabled or unconfigured, the orchestrator behaves exactly как
	// раніше (claude uses its constructor-time primary).
	var classifier *router.Classifier
	if os.Getenv("ROUTER_ENABLED") == "true" {
		cls, cerr := router.NewClassifier()
		if cerr != nil {
			log.Printf("WARN: router classifier init failed (%v), continuing без router", cerr)
		}
		classifier = cls
		if classifier == nil {
			log.Printf("router: ROUTER_ENABLED=true but ROUTER_CLASSIFIER_BASE_URL / _MODEL missing, router disabled")
		}
	}

	pending := elicit.NewPendingRegistry()

	// Shared MCP client used by the sync /messages handler. Its Elicitation
	// handler uses the policy fallback — no peer SSE stream to ask. For
	// /messages:stream we spin up a per-request session whose Elicitation
	// handler captures the request's stream by closure.
	mc, err := mcpclient.New(ctx, claude)
	if err != nil {
		return fmt.Errorf("init mcp client: %w", err)
	}
	defer mc.Close()
	log.Printf("mcp client connected: %d tools available, sampling+elicitation capabilities advertised", len(mc.Tools()))

	// Auto-discover kagent A2A peers by listing Agent CRDs у kagent ns
	// + fetching each peer's Agent Card. Replaces the hardcoded peer
	// table that used to live in internal/a2a/peer.go.
	discovery, err := peer.NewDiscovery(envOr("KAGENT_NAMESPACE", "kagent"))
	if err != nil {
		log.Printf("WARN: peer discovery init failed (%v) — tool list will be empty until in-cluster SA is reachable", err)
	} else {
		if err := discovery.Refresh(ctx); err != nil {
			log.Printf("WARN: initial peer discovery refresh failed: %v", err)
		}
		go discovery.Run(ctx, 5*time.Minute)
	}

	handler := &a2a.Handler{Claude: claude, MCP: mc, Discovery: discovery, Classifier: classifier}
	streamHandler := &a2a.StreamingHandler{Handler: handler, Pending: pending}
	respondHandler := &a2a.RespondHandler{Pending: pending}

	// A2A discovery — the Agent Card. Per RFC 8615 well-known URIs, this is
	// the canonical entry point for peers to learn about the agent.
	mux.HandleFunc("/.well-known/agent-card.json", agentcard.Handler())

	// Browser playground for interactive A2A testing — single-page HTML з
	// embedded form + JS, POSTs до /messages. Same-origin so no CORS.
	mux.HandleFunc("/ui", a2a.PlaygroundHandler)
	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		// Friendly root: redirect bare browser visits to /ui rather than 404.
		if r.URL.Path == "/" && r.Method == http.MethodGet {
			http.Redirect(w, r, "/ui", http.StatusFound)
			return
		}
		http.NotFound(w, r)
	})

	// A2A SendMessage (sync) — drives the Claude tool-use loop. Elicit
	// requests during execution fall back to policy decisions (no SSE
	// channel to push prompts on).
	mux.HandleFunc("/messages", handler.SendMessage)

	// A2A SendStreamingMessage — same agent loop, but emits SSE events as
	// the work progresses and bridges MCP elicits to the peer via
	// "input-required" events.
	mux.HandleFunc("/messages:stream", streamHandler.SendStreamingMessage)

	// Companion endpoint — peer POSTs here to satisfy an elicit it saw on
	// the SSE stream. Body: {correlation_id, action, content}.
	mux.HandleFunc("/messages:respond", respondHandler.Respond)

	// Liveness probe — used by K8s readiness/liveness probes if deployed.
	mux.HandleFunc("/healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	})

	srv := &http.Server{
		Addr:              *httpAddr,
		Handler:           extractTraceContext(logMiddleware(mux)),
		ReadHeaderTimeout: 10 * time.Second,
		WriteTimeout:      600 * time.Second, // long enough for multi-agent team coordination (up to 4 peer delegations + final synthesis)
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

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
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
