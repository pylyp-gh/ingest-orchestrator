// Package tracing wires OTel SDK to OTLP gRPC export з Phoenix, plus
// helpers that mirror doc-writer-mcp's setKind / endWithErr pattern.
// Two convention layers стack on each emitted span:
//
//   - OTel SemConv (gen_ai.*, http.*) — cross-vendor portability,
//     consumed by Tempo/Jaeger/Datadog with their respective views.
//   - OpenInference (openinference.span.kind, llm.*, etc.) — Phoenix-native
//     overlay that drives the "Kind" facet і specialised drilldown panels.
//
// Both are plain OTel attributes — adding OpenInference doesn't break
// OTel-only readers, just enriches the data for Phoenix.
package tracing

import (
	"context"
	"fmt"
	"os"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/codes"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	"go.opentelemetry.io/otel/sdk/resource"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	semconv "go.opentelemetry.io/otel/semconv/v1.26.0"
	"go.opentelemetry.io/otel/trace"
)

const tracerName = "github.com/pylyp-gh/ingest-orchestrator"

// OpenInference span kinds. Phoenix UI uses ці values to bucket spans
// into specialised drilldown panels (LLM token usage, retrieval scores,
// guardrail hit-rate, etc.).
const (
	KindChain     = "CHAIN"     // multi-step orchestration (agent loop, request handler)
	KindLLM       = "LLM"       // LLM completion / sampling call
	KindEmbedding = "EMBEDDING" // text → vector
	KindRetriever = "RETRIEVER" // vector DB query / write
	KindGuardrail = "GUARDRAIL" // validation / safety gate before LLM
	KindAgent     = "AGENT"     // agent / sub-agent invocation (delegation, A2A peer)
	KindTool      = "TOOL"      // generic tool / human-in-loop interaction
)

// Init wires the global OTel TracerProvider з an OTLP gRPC exporter
// pointed at the endpoint у OTEL_EXPORTER_OTLP_ENDPOINT. If that env
// var is empty, returns a no-op shutdown func — useful for local dev
// without Phoenix available.
//
// Returns a shutdown func that the caller must invoke у defer (flushes
// the batch span processor before process exit).
func Init(ctx context.Context, serviceName, version string) (func(context.Context) error, error) {
	endpoint := os.Getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
	if endpoint == "" {
		return func(context.Context) error { return nil }, nil
	}

	exp, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, fmt.Errorf("otlp grpc exporter: %w", err)
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(serviceName),
			semconv.ServiceVersion(version),
		),
	)
	if err != nil {
		return nil, fmt.Errorf("resource: %w", err)
	}

	tp := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exp),
		sdktrace.WithResource(res),
	)
	otel.SetTracerProvider(tp)
	// W3C TraceContext + Baggage — matches doc-writer-mcp's propagator
	// so traceparent flows seamlessly between processes.
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	return tp.Shutdown, nil
}

// Tracer returns the package-scoped tracer. All spans emitted from
// орchestrator code share this tracer name so Phoenix UI groups them
// together по telemetry.sdk.* resource.
func Tracer() trace.Tracer {
	return otel.Tracer(tracerName)
}

// SetKind attaches the OpenInference span.kind attribute.
func SetKind(span trace.Span, kind string) {
	span.SetAttributes(attribute.String("openinference.span.kind", kind))
}

// EndWithErr finalises a span, marking status=Error when err != nil
// and recording the error як an event. Use в defer:
//
//	ctx, span := tracing.Tracer().Start(ctx, "step.name")
//	defer func() { tracing.EndWithErr(span, err) }()
func EndWithErr(span trace.Span, err error) {
	if err != nil {
		span.SetStatus(codes.Error, err.Error())
		span.RecordError(err)
	}
	span.End()
}
