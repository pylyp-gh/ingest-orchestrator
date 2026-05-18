// Package a2a — peer.go implements an A2A peer client + the
// `delegate_to_kagent_peer` MCP-style meta-tool that lets the
// orchestrator's Claude delegate any subtask to one of the kagent
// agents available in the cluster.
//
// Why a meta-tool instead of just adding peer's tools to the
// orchestrator's tool list:
//   - kagent agents are A2A peers, not MCP servers. They speak
//     JSON-RPC `message/send` with Agent Card discovery at
//     /.well-known/agent-card.json, not the MCP `tools/call` protocol.
//   - Delegation through A2A preserves the peer's session context
//     (kagent_session_id) and its own internal capability negotiation
//     with its tools. Demonstrates real a2a coordination.
//   - Claude sees one tool with a clear "delegate to peer X" semantic
//     plus an agent_name enum — easy to plan multi-agent workflows.
//
// Lab 4 макс scope — multi-agent team. The orchestrator's Claude
// decomposes a single user request into multiple delegations across
// different kagent agents (writer for ingest, k8s for cluster queries,
// helm for chart questions, etc.), aggregating results into one reply.
package a2a

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"time"

	"github.com/google/uuid"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
)

// PeerToolName is the OpenAI ChatCompletion tool name Claude sees and
// invokes when it wants to delegate to any kagent A2A peer.
const PeerToolName = "delegate_to_kagent_peer"

// kagentPeers — known A2A peers in the cluster, surfaced to Claude as
// the enum of agent_name. Each entry describes the peer's domain so
// Claude can route subtasks to the right specialist. Endpoints resolve
// via in-cluster DNS (writer-agent.kagent.svc.cluster.local:8080).
//
// To add a peer: append here, redeploy. No system prompt change needed
// — the description below is part of the tool definition Claude reads.
var kagentPeers = []struct {
	Name        string
	Description string
}{
	{"writer-agent", "Documentation ingest — adds text to the Qdrant vector DB via the L0-L5 defence pipeline. Use for 'add this knowledge', 'store this doc'."},
	{"doc-agent", "Documentation query — reads from Qdrant. Use for 'find docs about X', 'what do we have on Y'."},
	{"k8s-agent", "Kubernetes cluster operations — kubectl-equivalent queries about resources, pods, services, deployments. Use for 'show me the running pods', 'what's the state of X', 'describe Y'."},
	{"helm-agent", "Helm chart operations — list releases, inspect values, troubleshoot installs. Use for 'what charts are installed', 'check release X status'."},
	{"istio-agent", "Istio service mesh queries — virtual services, gateways, destination rules. Use for service-mesh diagnostics."},
	{"observability-agent", "Logs + metrics inspection — Prometheus queries, log searches. Use for 'find errors in X', 'show CPU for Y'."},
	{"promql-agent", "PromQL query construction + execution. Use when user asks for a specific metric or wants help writing a query."},
}

// peerEndpoint computes the in-cluster URL for the named kagent agent.
// All kagent Agent CRDs reconcile to a Service of the same name in the
// `kagent` namespace listening on :8080 — pattern owned by kagent.
func peerEndpoint(agentName string) string {
	// Allow per-deploy override only for writer-agent (legacy env, kept
	// for backward compatibility with earlier Lab 4 макс iteration).
	if agentName == "writer-agent" {
		if v := os.Getenv("WRITER_AGENT_A2A_URL"); v != "" {
			return v
		}
	}
	return fmt.Sprintf("http://%s.kagent.svc.cluster.local:8080/", agentName)
}

// peerToolDefinition builds the OpenAI tool description Claude sees.
// Enum + per-peer description drives task routing. Made a builder
// rather than a package var to avoid map-aliasing surprises.
func peerToolDefinition() openai.ChatCompletionToolParam {
	enumValues := make([]string, len(kagentPeers))
	descriptions := "Available peers:\n"
	for i, p := range kagentPeers {
		enumValues[i] = p.Name
		descriptions += fmt.Sprintf("  - %s: %s\n", p.Name, p.Description)
	}

	return openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{
			Name: PeerToolName,
			Description: openai.String(
				"Delegate a subtask to a kagent A2A peer agent. Pick the agent_name " +
					"based on the request domain. You can call this tool multiple times " +
					"in one conversation to decompose a complex request across several " +
					"specialised peers, then synthesise their replies into a final answer.\n\n" +
					descriptions),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"agent_name": map[string]any{
						"type":        "string",
						"enum":        enumValues,
						"description": "Which kagent peer to invoke. Choose based on the subtask's domain (see peer descriptions above).",
					},
					"text": map[string]any{
						"type":        "string",
						"description": "The full natural-language request for the chosen peer. Frame it as a complete user request so the peer has all the context it needs to act independently.",
					},
				},
				"required": []string{"agent_name", "text"},
			},
		},
	}
}

// jsonRPCRequest mirrors the A2A spec's `message/send` envelope.
type jsonRPCRequest struct {
	JSONRPC string         `json:"jsonrpc"`
	Method  string         `json:"method"`
	Params  map[string]any `json:"params"`
	ID      string         `json:"id"`
}

// jsonRPCResponse — minimal subset; we only extract artifacts[].parts[].text.
type jsonRPCResponse struct {
	JSONRPC string `json:"jsonrpc"`
	ID      string `json:"id"`
	Result  *struct {
		Artifacts []struct {
			ArtifactID string `json:"artifactId"`
			Parts      []struct {
				Kind string `json:"kind"`
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"artifacts"`
		ContextID string `json:"contextId"`
		Status    struct {
			State string `json:"state"`
		} `json:"status"`
	} `json:"result,omitempty"`
	Error *struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// invokeKagentPeer sends an A2A message/send JSON-RPC call to the
// named kagent agent and returns the concatenated artifact text.
// Caller-side timeout = 180s — doc-agent / observability-agent paths
// can chain Qdrant cosine search + Ollama embeddings + multiple Claude
// completions, totalling 90-120s. Generous bound prevents premature
// retries; the upstream orchestrator already has its own request-level
// budget to terminate stuck delegations.
func invokeKagentPeer(ctx context.Context, agentName, userText string) (string, error) {
	endpoint := peerEndpoint(agentName)
	req := jsonRPCRequest{
		JSONRPC: "2.0",
		Method:  "message/send",
		Params: map[string]any{
			"message": map[string]any{
				"role":      "user",
				"messageId": uuid.NewString(),
				"parts": []map[string]any{
					{"kind": "text", "text": userText},
				},
			},
		},
		ID: uuid.NewString(),
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", fmt.Errorf("peer: marshal request: %w", err)
	}

	callCtx, cancel := context.WithTimeout(ctx, 180*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(callCtx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("peer: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	log.Printf("[peer] POST %s (agent=%s) id=%s", endpoint, agentName, req.ID)
	resp, err := http.DefaultClient.Do(httpReq)
	if err != nil {
		return "", fmt.Errorf("peer: do: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("peer: read: %w", err)
	}
	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("peer: HTTP %d: %s", resp.StatusCode, truncatePeer(string(respBody), 200))
	}

	var parsed jsonRPCResponse
	if err := json.Unmarshal(respBody, &parsed); err != nil {
		return "", fmt.Errorf("peer: parse response: %w (raw: %s)", err, truncatePeer(string(respBody), 200))
	}
	if parsed.Error != nil {
		return "", fmt.Errorf("peer: JSON-RPC error %d: %s", parsed.Error.Code, parsed.Error.Message)
	}
	if parsed.Result == nil {
		return "", fmt.Errorf("peer: response has no result")
	}

	var text string
	for _, a := range parsed.Result.Artifacts {
		for _, p := range a.Parts {
			if p.Kind == "text" {
				text += p.Text
			}
		}
	}
	if text == "" {
		return "", fmt.Errorf("peer: no text artifact in response (state=%s)", parsed.Result.Status.State)
	}
	log.Printf("[peer] result: state=%s ctx=%s textChars=%d", parsed.Result.Status.State, parsed.Result.ContextID, len(text))
	return text, nil
}

func truncatePeer(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
