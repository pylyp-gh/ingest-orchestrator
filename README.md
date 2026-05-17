# ingest-orchestrator

A2A (Agent-to-Agent) protocol agent that orchestrates documentation ingest into
Qdrant via the [doc-writer-mcp](https://github.com/pylyp-gh/doc-writer-mcp)
server. Demonstrates a **full-capability MCP client** integrated with the
Anthropic Claude API: declares Sampling and Elicitation capabilities and bridges
them to external A2A peers.

Built for Lab 4 of the
[kagent-flux-lab](https://github.com/pylyp-gh/kagent-flux-lab) GitOps lab.
Companion piece to writer-agent (kagent-managed, tools-only runtime) — together
they show the spectrum of MCP clients from limited to full-capability.

## Endpoints (A2A protocol)

| Method | Path                           | Purpose                       |
| ------ | ------------------------------ | ----------------------------- |
| GET    | `/.well-known/agent-card.json` | A2A discovery (RFC 8615)      |
| POST   | `/messages`                    | SendMessage — task initiation |
| GET    | `/healthz`                     | Liveness probe                |

## Phases

| Phase | Scope                                            | Status |
| ----- | ------------------------------------------------ | ------ |
| 1     | HTTP server + Agent Card + echo skill            | DONE   |
| 2     | MCP client integration → doc-writer.add_document | TODO   |
| 3     | Sampling bridge → Anthropic API                  | TODO   |
| 4     | Elicitation bridge → A2A streaming               | TODO   |

## Run locally

```bash
go run cmd/orchestrator/main.go --http :8080
curl http://localhost:8080/.well-known/agent-card.json | jq .
curl -X POST http://localhost:8080/messages \
  -H 'Content-Type: application/json' \
  -d '{"message":{"role":"user","content":[{"type":"text","text":"hello"}]}}'
```

## Build

`kmcp build` not applicable (not an MCP server). Standard Docker:

```bash
docker build -t ingest-orchestrator:dev .
```

CI/CD via GitHub Actions publishes multi-arch images to GHCR on push to main.
