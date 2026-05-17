// Package mcpclient wraps the modelcontextprotocol/go-sdk client for use by
// the agent loop. Establishes a Streamable HTTP session to a target MCP
// server (doc-writer-mcp у Phase 2), exposes ListTools and CallTool, and
// keeps the session alive for the lifetime of the orchestrator process.
//
// Phase 2 declares NO sampling/elicitation capabilities — server falls back
// to its built-in defaults (auto-create collection, auto-decline duplicates).
// Phase 3 adds Sampling bridge → Anthropic. Phase 4 adds Elicitation bridge
// → A2A streaming back to the calling peer.
package mcpclient

import (
	"context"
	"fmt"
	"os"

	"github.com/modelcontextprotocol/go-sdk/mcp"
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
// a ready-to-use client.
func New(ctx context.Context) (*Client, error) {
	url := envOr("MCP_SERVER_URL", "https://doc-writer.ash.ph.lab/mcp")
	impl := &mcp.Implementation{
		Name:    "ingest-orchestrator",
		Version: "0.1.0",
	}
	mcpClient := mcp.NewClient(impl, nil)
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
