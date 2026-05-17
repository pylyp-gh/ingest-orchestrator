// Package mcpclient wraps the modelcontextprotocol/go-sdk client for use by
// the agent loop. Establishes a Streamable HTTP session to a target MCP
// server (doc-writer-mcp у Phase 2), exposes ListTools and CallTool, and
// keeps the session alive for the lifetime of the orchestrator process.
//
// Phase 3 adds a Sampling bridge — when the MCP server issues a
// sampling/createMessage request (e.g. doc-writer-mcp's L5 verdict gate),
// the registered CreateMessageHandler translates SamplingMessages into an
// OpenAI ChatCompletion call against the same agentgateway-proxy used by
// the agent loop. Capability is advertised automatically by the SDK once
// the handler field is set.
//
// Phase 4 will add an Elicitation bridge → A2A streaming back to the
// calling peer.
package mcpclient

import (
	"context"
	"fmt"
	"log"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
)

type Client struct {
	session *mcp.ClientSession
	tools   []*mcp.Tool
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// New connects to the configured MCP server URL via Streamable HTTP,
// performs the initialize handshake, lists available tools, and returns
// a ready-to-use client. Registers a CreateMessageHandler so the SDK
// auto-advertises sampling capability during the handshake.
func New(ctx context.Context, claude *llm.Claude) (*Client, error) {
	url := envOr("MCP_SERVER_URL", "https://doc-writer.ash.ph.lab/mcp")
	impl := &mcp.Implementation{
		Name:    "ingest-orchestrator",
		Version: "0.1.0",
	}
	opts := &mcp.ClientOptions{
		CreateMessageHandler: buildSamplingHandler(claude),
	}
	mcpClient := mcp.NewClient(impl, opts)
	transport := &mcp.StreamableClientTransport{Endpoint: url}
	session, err := mcpClient.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp connect %s: %w", url, err)
	}
	// List tools once at startup — keeps the agent loop fast and avoids
	// repeating the round-trip on every call. Tools rarely change at runtime.
	listResp, err := session.ListTools(ctx, &mcp.ListToolsParams{})
	if err != nil {
		_ = session.Close()
		return nil, fmt.Errorf("list tools: %w", err)
	}
	return &Client{session: session, tools: listResp.Tools}, nil
}

func (c *Client) Tools() []*mcp.Tool { return c.tools }

// CallTool invokes a tool by name with the given JSON-encodable arguments.
// Returns the raw CallToolResult — the caller is responsible for parsing
// content blocks (text, image, resource, etc.).
func (c *Client) CallTool(ctx context.Context, name string, args map[string]any) (*mcp.CallToolResult, error) {
	resp, err := c.session.CallTool(ctx, &mcp.CallToolParams{
		Name:      name,
		Arguments: args,
	})
	if err != nil {
		return nil, fmt.Errorf("call tool %s: %w", name, err)
	}
	return resp, nil
}

func (c *Client) Close() error {
	if c == nil || c.session == nil {
		return nil
	}
	return c.session.Close()
}

// buildSamplingHandler returns a CreateMessageHandler closed over the given
// Claude wrapper. The handler translates an MCP sampling/createMessage
// request into an OpenAI ChatCompletion call against agentgateway, then
// wraps the assistant text back into a CreateMessageResult.
//
// Translation contract:
//   - params.SystemPrompt → openai.SystemMessage (prepended)
//   - params.Messages[].Role "user" → openai.UserMessage
//   - params.Messages[].Role "assistant" → openai.AssistantMessage
//   - params.Messages[].Content → only *TextContent supported in Phase 3
//
// Tools deliberately omitted: doc-writer-mcp's L5 calls are single-turn
// classification/extraction — no tool use needed. CreateMessageWithToolsHandler
// (SDK variant) would be the place to wire tools if a future server requests
// them; ignored here to keep the handler minimal.
func buildSamplingHandler(claude *llm.Claude) func(context.Context, *mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
	return func(ctx context.Context, req *mcp.CreateMessageRequest) (*mcp.CreateMessageResult, error) {
		params := req.Params
		if params == nil {
			return nil, fmt.Errorf("sampling: nil params")
		}

		msgs := make([]openai.ChatCompletionMessageParamUnion, 0, len(params.Messages)+1)
		if params.SystemPrompt != "" {
			msgs = append(msgs, openai.SystemMessage(params.SystemPrompt))
		}
		for i, m := range params.Messages {
			tc, ok := m.Content.(*mcp.TextContent)
			if !ok {
				return nil, fmt.Errorf("sampling: message %d content type %T unsupported (Phase 3 = text only)", i, m.Content)
			}
			switch m.Role {
			case mcp.Role("user"):
				msgs = append(msgs, openai.UserMessage(tc.Text))
			case mcp.Role("assistant"):
				msgs = append(msgs, openai.AssistantMessage(tc.Text))
			default:
				return nil, fmt.Errorf("sampling: message %d role %q unsupported", i, m.Role)
			}
		}

		log.Printf("[sampling] in: %d messages, system=%t, maxTok=%d", len(params.Messages), params.SystemPrompt != "", params.MaxTokens)
		assistant, err := claude.Complete(ctx, msgs, nil)
		if err != nil {
			return nil, fmt.Errorf("sampling: claude complete: %w", err)
		}

		log.Printf("[sampling] out: %d chars", len(assistant.Content))
		return &mcp.CreateMessageResult{
			Content:    &mcp.TextContent{Text: assistant.Content},
			Model:      claude.Model(),
			Role:       mcp.Role("assistant"),
			StopReason: "endTurn",
		}, nil
	}
}
