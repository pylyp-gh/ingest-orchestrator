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
)

// maxIterations bounds the tool-use loop to prevent runaway behaviour
// (e.g., Claude calling the same tool in a cycle). Each iteration is one
// model round-trip + at most N tool calls.
const maxIterations = 8

const systemPrompt = `You are the Ingest Orchestrator, an A2A agent that helps users add documentation to a Qdrant vector database. You have two ways to handle ingest requests:

1. add_document (MCP tool, local) — call doc-writer-mcp yourself. Use when the user provides direct text and you want fine-grained control: see the action (inserted/replaced/versioned), point ID, and metadata in the tool result.

2. delegate_to_writer_agent (A2A peer) — hand the whole ingest off to writer-agent, a kagent A2A peer that drives the same L0-L5 defence + Sampling + Elicitation pipeline end-to-end. Use when the user asks something open-ended ("add this", "store this knowledge") and you want writer-agent to own the full flow. You receive writer-agent's final natural-language response.

Choosing between them: prefer add_document for crisp ingest tasks where you can frame the call cleanly; prefer delegate_to_writer_agent when the request is more conversational or you want to demonstrate cross-agent delegation. If unsure, default to add_document.

For both tools:
- Pass the user's text verbatim. Do not rewrite, summarise, or paraphrase — the quality gates downstream depend on the original text.
- If the user passes a URL alongside the text, include it as sourceUrl (add_document only; delegate_to_writer_agent will detect URLs in the natural-language text).
- If the tool returns a soft error or rejection, explain why and suggest what the user can do.
- If the user asks something that isn't an ingest request, briefly explain what you do and ask them to provide text to ingest.
- Reply in the language of the user.`

// Loop runs the agent loop on the given user text, using the given Claude
// and MCP clients. Returns the terminal text response. The tool list
// presented to Claude is the union of MCP tools (add_document via
// doc-writer-mcp) plus the delegate_to_writer_agent A2A meta-tool.
func Loop(ctx context.Context, claude *llm.Claude, mc *mcpclient.Client, userText string) (string, error) {
	tools := mcpToolsToOpenAI(mc.Tools())
	tools = append(tools, peerToolDefinition())

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
		text, _ := args["text"].(string)
		if text == "" {
			return "error: delegate_to_writer_agent requires a non-empty 'text' argument"
		}
		out, err := invokeWriterAgent(ctx, text)
		if err != nil {
			return fmt.Sprintf("error from peer: %v", err)
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
