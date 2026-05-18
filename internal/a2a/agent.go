// Package a2a — agent.go implements the Claude tool-use loop.
//
// Flow:
//   1. Convert MCP tools list into OpenAI ChatCompletion tool definitions.
//   2. Send system prompt + user message + tools to Claude (via gateway).
//   3. If Claude returns plain text → that's the terminal response.
//   4. If Claude returns tool_calls → for each, invoke the MCP server tool,
//      feed the result back as a "tool" role message, recurse.
//   5. Loop until plain text or maxIterations hit (safety bound).
//
// Phase 2 keeps the loop synchronous. Phase 3-4 add Sampling/Elicitation
// bridges so the MCP server can call back into the orchestrator mid-flight.
package a2a

import (
	"context"
	"encoding/json"
	"fmt"
	"log"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
	"github.com/pylyp-gh/ingest-orchestrator/internal/peer"
)

// maxIterations bounds the tool-use loop to prevent runaway behaviour
// (e.g., Claude calling the same tool in a cycle). Each iteration is one
// model round-trip + at most N tool calls.
const maxIterations = 8

const systemPrompt = `You are the Ingest Orchestrator, an A2A team coordinator. You decompose user requests into subtasks and route them to either your own MCP tool (add_document) or to specialised kagent A2A peers via delegate_to_kagent_peer.

Your toolbelt:

1. add_document (MCP tool, local) — call doc-writer-mcp yourself. Use when the user provides direct text and you want fine-grained control (you see the action taken, point ID, metadata in the tool result).

2. delegate_to_kagent_peer (A2A) — invoke any kagent peer agent by name. See the tool description for the list of available peers and what each does. Use this to:
   - Hand off a single subtask to a specialist (e.g. k8s-agent for cluster queries)
   - Decompose a complex multi-domain request into parallel delegations (multiple tool_calls in one turn) and aggregate the results yourself

Team-coordination guidance:
- Read the user's request and identify how many distinct domains it touches (docs ingest, cluster inspection, helm releases, observability, mesh diagnostics).
- For each distinct domain, choose the right peer from delegate_to_kagent_peer's enum.
- If the request is single-domain, just invoke one tool/peer.
- If the request spans multiple domains, fire multiple delegate_to_kagent_peer calls in one turn (tool_choice will execute them all), then synthesise the peer replies into a single coherent answer for the user.
- Don't re-summarise inside add_document or delegate_to_kagent_peer calls — pass the user's text verbatim. Quality gates and the peers' own LLMs depend on the original phrasing.

If the request is purely about ingesting one document, you can use add_document directly (faster, no extra hop). If you want a writer-agent demo or the request is conversational ("save this thought"), delegate.

Reply in the language of the user. If a peer returns an error, explain what happened and suggest a recovery (e.g. "retry without the URL", "shorten the text", "the L5 gate is currently blocking — try ENABLE_SAMPLING=false on the doc-writer-mcp server").`

// Loop runs the agent loop on the given user text, using the given Claude
// and MCP clients. Returns the terminal text response. The tool list
// presented to Claude is the union of MCP tools (add_document via
// doc-writer-mcp) plus the delegate_to_kagent_peer A2A meta-tool whose
// enum is built from the live kagent peer list.
func Loop(ctx context.Context, claude *llm.Claude, mc *mcpclient.Client, discovery *peer.Discovery, userText string) (string, error) {
	tools := mcpToolsToOpenAI(mc.Tools())
	var peers []peer.Peer
	if discovery != nil {
		peers = discovery.Peers()
	}
	tools = append(tools, peerToolDefinition(peers))

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
		openai.UserMessage(userText),
	}

	for i := 0; i < maxIterations; i++ {
		assistant, err := claude.Complete(ctx, messages, tools)
		if err != nil {
			return "", fmt.Errorf("claude iteration %d: %w", i, err)
		}

		// Terminal text — no tool calls → return immediately.
		if len(assistant.ToolCalls) == 0 {
			return assistant.Content, nil
		}

		// Append assistant turn (with tool_calls) before sending tool results.
		messages = append(messages, assistant.ToParam())

		// Execute each tool call serially, append result messages.
		for _, call := range assistant.ToolCalls {
			result := executeToolCall(ctx, mc, call)
			messages = append(messages, openai.ToolMessage(result, call.ID))
		}
	}

	return "", fmt.Errorf("max iterations (%d) exceeded — Claude may be looping on tool calls", maxIterations)
}

// executeToolCall invokes the named MCP tool with the JSON arguments from
// Claude. Returns a textual result string suitable for feeding back as a
// "tool" role message. Errors are surfaced as text so Claude can see them
// and adjust strategy (e.g., "duplicate exists — ask user").
func executeToolCall(ctx context.Context, mc *mcpclient.Client, call openai.ChatCompletionMessageToolCall) string {
	name := call.Function.Name
	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		return fmt.Sprintf("error: failed to parse tool arguments JSON: %v (raw: %q)", err, call.Function.Arguments)
	}
	log.Printf("[agent] tool call: %s args=%s", name, call.Function.Arguments)

	// Dispatch — A2A peer delegation routes outside MCP entirely.
	if name == PeerToolName {
		agentName, _ := args["agent_name"].(string)
		text, _ := args["text"].(string)
		if agentName == "" || text == "" {
			return "error: delegate_to_kagent_peer requires non-empty 'agent_name' and 'text' arguments"
		}
		out, err := invokeKagentPeer(ctx, agentName, text)
		if err != nil {
			return fmt.Sprintf("error from peer %s: %v", agentName, err)
		}
		return out
	}

	resp, err := mc.CallTool(ctx, name, args)
	if err != nil {
		return fmt.Sprintf("error: tool call failed: %v", err)
	}
	// Concatenate text content blocks from the result.
	var out string
	for _, c := range resp.Content {
		if tc, ok := c.(*mcp.TextContent); ok {
			out += tc.Text + "\n"
		}
	}
	if out == "" {
		out = "(tool returned no text content)"
	}
	if resp.IsError {
		out = "error from tool: " + out
	}
	return out
}

// mcpToolsToOpenAI converts MCP tool definitions into OpenAI ChatCompletion
// tool definitions. Schema is JSON-Schema in both worlds; the only mapping
// is field-name conventions (mcp.InputSchema → openai parameters).
func mcpToolsToOpenAI(tools []*mcp.Tool) []openai.ChatCompletionToolParam {
	out := make([]openai.ChatCompletionToolParam, 0, len(tools))
	for _, t := range tools {
		var params map[string]any
		if t.InputSchema != nil {
			b, _ := json.Marshal(t.InputSchema)
			_ = json.Unmarshal(b, &params)
		}
		out = append(out, openai.ChatCompletionToolParam{
			Function: shared.FunctionDefinitionParam{
				Name:        t.Name,
				Description: openai.String(t.Description),
				Parameters:  params,
			},
		})
	}
	return out
}
