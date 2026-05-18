// Package a2a — peer.go implements an A2A peer client + the
// `delegate_to_writer_agent` MCP-style meta-tool that lets the
// orchestrator's Claude delegate document-ingest work to the kagent
// writer-agent.
//
// Why a meta-tool instead of just adding writer-agent's MCP tools to
// the orchestrator's tool list:
//   - writer-agent is an A2A peer, not an MCP server. It speaks
//     JSON-RPC `message/send` with Agent Card discovery at
//     /.well-known/agent-card.json, not the MCP `tools/call` protocol.
//   - Delegation through A2A preserves writer-agent's session context
//     (kagent_session_id) and its own Sampling/Elicitation negotiation
//     with doc-writer-mcp. Going through this peer rather than
//     calling doc-writer-mcp directly demonstrates real a2a coordination.
//   - Claude sees a single tool with a clear "delegate" semantic rather
//     than a flat tool inventory mixing local + remote concerns.
//
// Lab 4 макс scope — single peer (writer-agent). Generalising to N peers
// = parameterise URL + tool name; trivial future extension.
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
// invokes when it wants to delegate to writer-agent.
const PeerToolName = "delegate_to_writer_agent"

// peerEndpointEnv lets the deploy override the in-cluster default. Empty
// → fallback to writer-agent.kagent.svc.cluster.local:8080. The lab's
// public hostname (https://writer-agent.ash.ph.lab/) also works through
// agentgateway but in-cluster DNS is faster (no TLS, no proxy hop).
const peerEndpointEnv = "WRITER_AGENT_A2A_URL"

func peerEndpoint() string {
	if v := os.Getenv(peerEndpointEnv); v != "" {
		return v
	}
	return "http://writer-agent.kagent.svc.cluster.local:8080/"
}

// peerToolDefinition is the OpenAI tool description Claude sees. Made a
// builder rather than a package var to avoid map-aliasing surprises.
func peerToolDefinition() openai.ChatCompletionToolParam {
	return openai.ChatCompletionToolParam{
		Function: shared.FunctionDefinitionParam{
			Name: PeerToolName,
			Description: openai.String(
				"Delegate a document ingestion task to the writer-agent A2A peer. " +
					"Use this when the user asks to add documentation and you want " +
					"writer-agent to drive the L0-L5 defence pipeline + Elicitation + " +
					"Sampling end-to-end. Returns writer-agent's final response text " +
					"(in the user's language). Use add_document tool if you prefer to " +
					"handle the ingest yourself instead of delegating."),
			Parameters: map[string]any{
				"type": "object",
				"properties": map[string]any{
					"text": map[string]any{
						"type":        "string",
						"description": "The full natural-language request to send to writer-agent. Usually the user's original text plus any framing needed (e.g. 'Add this document: ...'). Pass verbatim if possible.",
					},
				},
				"required": []string{"text"},
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

// invokeWriterAgent sends an A2A message/send JSON-RPC call to
// writer-agent and returns the concatenated artifact text. Caller-side
// timeout = 90s (writer-agent's full pipeline = MCP CallTool +
// Sampling x2 + final Claude completion = up to 60-70s in practice).
func invokeWriterAgent(ctx context.Context, userText string) (string, error) {
	endpoint := peerEndpoint()
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

	callCtx, cancel := context.WithTimeout(ctx, 90*time.Second)
	defer cancel()

	httpReq, err := http.NewRequestWithContext(callCtx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return "", fmt.Errorf("peer: new request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Accept", "application/json")

	log.Printf("[peer] POST %s id=%s", endpoint, req.ID)
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
