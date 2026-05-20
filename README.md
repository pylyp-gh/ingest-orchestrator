# ingest-orchestrator

A2A (Agent-to-Agent) team coordinator з full-capability MCP client integration
and multi-agent task delegation. Drives a Claude tool-use loop that decomposes
incoming requests into subtasks і routes each to the right tool:

- **Local MCP tool** — `add_document` via doc-writer-mcp (Sampling-gated
  document ingest into Qdrant).
- **A2A peer delegation** — `delegate_to_kagent_peer` invokes any kagent Agent
  on its native JSON-RPC `/messages` endpoint, auto-discovered from the
  cluster's kagent Agent CRDs.

Despite the historical name, ingest is now one of several capabilities — the
agent is general enough to fan out a multi-domain request (e.g. "diagnose
cluster health AND save the analysis as a doc") across heterogeneous peers and
synthesise their replies into one coherent answer.

Companion piece to writer-agent (kagent-managed, tools-only runtime) and the
broader [kagent-flux-lab](https://github.com/pylyp-gh/kagent-flux-lab) GitOps
stack. Together вони illustrate the spectrum of MCP clients — from declarative
kagent agents до full-control custom Go runtimes.

## Capabilities

| Capability                      | Where                                                                                                                                                         |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| A2A protocol server             | `/messages`, `/messages:stream`, `/messages:respond`, `/.well-known/agent-card.json`                                                                          |
| Claude tool-use loop            | `internal/a2a/agent.go` (up to 8 iterations)                                                                                                                  |
| MCP Streamable HTTP client      | `internal/mcpclient/` — declares Sampling + Elicitation capabilities to the server                                                                            |
| Sampling bridge → Anthropic     | server's `sampling/createMessage` calls hop through the same OpenAI SDK + agentgateway pipeline as the agent's own LLM calls                                  |
| Elicitation bridge → A2A stream | server's `elicitation/create` calls surface to the calling peer як SSE `input-required` events; peer POSTs to `/messages:respond` to satisfy                  |
| Multi-agent peer delegation     | `delegate_to_kagent_peer` meta-tool з a live enum built from kagent Agent CRDs у `kagent` ns                                                                  |
| Peer auto-discovery             | `internal/peer/discovery.go` — informer-style refresh every 5 min                                                                                             |
| OTel tracing з OpenInference    | `internal/tracing/` — spans tagged з `openinference.span.kind` (AGENT/LLM/TOOL/CHAIN) so Phoenix renders proper drilldown panels                              |
| Ollama fallback on 4xx          | `internal/llm/claude.go` — on Anthropic 400/401/429 etc. falls back to a secondary OpenAI-compat endpoint (Ollama by default), preserves single-span identity |
| Playground UI                   | `/ui` — single-page HTML form для interactive testing                                                                                                         |

## Endpoints

| Method | Path                           | Purpose                                                                |
| ------ | ------------------------------ | ---------------------------------------------------------------------- |
| GET    | `/.well-known/agent-card.json` | A2A discovery (RFC 8615) — capabilities + skill list                   |
| GET    | `/ui`                          | Browser playground                                                     |
| POST   | `/messages`                    | Sync SendMessage — drives the Claude loop, returns a completed Task    |
| POST   | `/messages:stream`             | Streaming SendStreamingMessage — emits SSE events, bridges Elicitation |
| POST   | `/messages:respond`            | Peer-side response to an Elicitation event seen on the stream          |
| GET    | `/healthz`                     | Liveness probe                                                         |

## Tools exposed to Claude

| Tool name                 | Source         | When to use                                                                                                                     |
| ------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| `add_document`            | doc-writer-mcp | Direct text ingestion into Qdrant `doc-writer` collection з L0-L5 quality gates                                                 |
| `delegate_to_kagent_peer` | A2A meta-tool  | Hand off a subtask to a specialist kagent peer (k8s, helm, kmcp, istio, observability...) — enum built dynamically from cluster |

Multi-domain requests trigger several `delegate_to_kagent_peer` calls у one turn
(parallel tool execution), then the agent synthesises peer replies into a final
answer for the user.

## Trace tree (Phoenix / OpenInference)

```
orchestrator.loop                                  AGENT
├─ llm.chat.completions                            LLM     (×N — agent loop turns)
│    └─ llm.token_count.{prompt,completion,total}
│    └─ llm.fallback.used (when 4xx → Ollama)
├─ tool.call.add_document                          TOOL
│    └─ tool.add_document (doc-writer-mcp)         CHAIN   ← traceparent-stitched
│        ├─ validate.L0 / validate.L1              GUARDRAIL
│        ├─ qdrant.* / dedup.*                     RETRIEVER
│        ├─ sampling.verdict / sampling.metadata   LLM
│        ├─ embed.ollama                           EMBEDDING
│        └─ qdrant.upsert                          RETRIEVER
└─ tool.call.delegate_to_kagent_peer               AGENT
     └─ (downstream A2A peer's own loop)
```

W3C TraceContext propagates через MCP Streamable HTTP headers і through the A2A
JSON-RPC POST, so a Phoenix trace spans up to three processes (client →
orchestrator → doc-writer-mcp / kagent peer) under one trace_id.

## Environment variables

| Variable                      | Default                                                                | Purpose                                                                   |
| ----------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| `OPENAI_BASE_URL`             | `https://agentgateway-proxy.agentgateway-system.svc.cluster.local/v1/` | Primary LLM endpoint (OpenAI shape, agentgateway translates to Anthropic) |
| `OPENAI_API_KEY`              | `stub-gateway-injects-real-key`                                        | Placeholder — gateway substitutes the real key from `anthropic-secret`    |
| `MODEL_NAME`                  | `claude-opus-4-6`                                                      | Forwarded як `model` field у ChatCompletion requests                      |
| `MCP_SERVER_URL`              | `https://doc-writer.ash.ph.lab/mcp`                                    | doc-writer-mcp Streamable HTTP endpoint                                   |
| `A2A_SERVICE_ENDPOINT`        | _required_                                                             | This agent's externally-reachable URL — included у the Agent Card         |
| `KAGENT_NAMESPACE`            | `kagent`                                                               | Namespace que `peer.Discovery` watches для kagent Agent CRDs              |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | _empty_ (tracing disabled)                                             | OTLP gRPC endpoint, e.g. `phoenix-svc.phoenix.svc.cluster.local:4317`     |
| `OLLAMA_FALLBACK_URL`         | _empty_ (fallback disabled)                                            | OpenAI-compat endpoint used coли primary returns 4xx                      |
| `OLLAMA_FALLBACK_MODEL`       | `qwen2.5:14b`                                                          | Model name passed to the fallback client                                  |
| `SSL_CERT_FILE`               | system trust store                                                     | Path to lab CA bundle when running in-cluster з `*.ash.ph.lab` certs      |

## Run locally

```bash
# Minimal smoke (no MCP, no Anthropic, no peers — just A2A scaffolding):
go run cmd/orchestrator/main.go --http :8080
curl http://localhost:8080/.well-known/agent-card.json | jq .
curl -X POST http://localhost:8080/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":{"role":"user","content":[{"type":"text","text":"hello"}]}}'
```

For a full run with peer discovery + MCP + tracing, set the env vars listed
above and connect to a live cluster (the deployment manifest у
[kagent-flux-lab](https://github.com/pylyp-gh/kagent-flux-lab/tree/main/clusters/kind-lab/apps/ingest-orchestrator)
shows the production wiring).

## Build

```bash
docker build -t ingest-orchestrator:dev .
```

CI/CD via GitHub Actions publishes multi-arch images (linux/amd64 + linux/arm64)
to `ghcr.io/pylyp-gh/ingest-orchestrator:sha-<commit>` on push to `main`. The
Flux deployment pins by commit SHA.

## Related

- [doc-writer-mcp](https://github.com/pylyp-gh/doc-writer-mcp) — the MCP server
  this agent calls into для document ingestion.
- [kagent-flux-lab](https://github.com/pylyp-gh/kagent-flux-lab) — the parent
  GitOps stack що deploys this agent alongside Phoenix, Qdrant, agentgateway,
  kagent, and the rest of the AI infra.
