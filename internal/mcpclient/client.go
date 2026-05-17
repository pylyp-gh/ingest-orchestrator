// Package mcpclient wraps the modelcontextprotocol/go-sdk client for use by
// the agent loop. Establishes a Streamable HTTP session to a target MCP
// server (doc-writer-mcp у Phase 2), exposes ListTools and CallTool, and
// keeps the session alive for the lifetime of the orchestrator process.
//
// Phase 3 adds a Sampling bridge — when the MCP server issues a
// sampling/createMessage request (e.g. doc-writer-mcp's L5 verdict gate),
// the registered CreateMessageHandler translates SamplingMessages into an
// OpenAI ChatCompletion call against the same agentgateway-proxy used by
// the agent loop.
//
// Phase 4 adds an Elicitation bridge. When the server issues
// elicitation/create, the handler looks for an SSE stream on the request
// context (placed there by the /messages:stream A2A handler) and pushes
// an "input-required" event to the peer. Then blocks on a channel from
// the PendingRegistry until the peer POSTs to /messages:respond with the
// matching correlation ID. Without an attached stream (e.g. plain
// /messages request) the handler falls back to a deterministic policy so
// the tool flow doesn't deadlock.
package mcpclient

import (
	"context"
	"fmt"
	"log"
	"os"

	"time"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go"
	"github.com/pylyp-gh/ingest-orchestrator/internal/elicit"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
)

// ElicitTimeout bounds how long the elicit bridge waits for a peer reply
// before falling back to "decline". Generous so a human peer can reason
// about the prompt, but short enough that orphaned MCP calls don't pile
// up if peer disconnected.
const ElicitTimeout = 2 * time.Minute

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
// a ready-to-use client. Registers both Sampling and Elicitation handlers
// so the SDK auto-advertises both capabilities during the handshake.
func New(ctx context.Context, claude *llm.Claude, pending *elicit.PendingRegistry) (*Client, error) {
	url := envOr("MCP_SERVER_URL", "https://doc-writer.ash.ph.lab/mcp")
	impl := &mcp.Implementation{
		Name:    "ingest-orchestrator",
		Version: "0.1.0",
	}
	opts := &mcp.ClientOptions{
		CreateMessageHandler: buildSamplingHandler(claude),
		ElicitationHandler:   buildElicitationHandler(pending),
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

// buildElicitationHandler returns an ElicitationHandler that bridges
// MCP elicit requests to the SSE stream attached to the calling context.
//
// Behaviour matrix:
//
//	stream present (streaming endpoint) → push "input-required" event,
//	                                       block on peer reply channel,
//	                                       return decoded ElicitResult.
//	stream absent (sync /messages call) → deterministic policy:
//	                                       bootstrap (create=bool) → accept create
//	                                       duplicate (choice=enum)  → decline
//	                                       other schemas            → decline
//
// Policy fallback rationale:
//   - Bootstrap is safe to auto-accept: empty Qdrant collection has no
//     content risk. Matches the doc-writer-mcp server-side default.
//   - Duplicate auto-decline avoids silent data loss. Caller (Claude
//     agent) sees the decline and can re-ask the user explicitly.
//
// Timeout: ElicitTimeout bounds the wait so a disconnected peer doesn't
// pin an MCP call forever.
func buildElicitationHandler(pending *elicit.PendingRegistry) func(context.Context, *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
	return func(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
		params := req.Params
		if params == nil {
			return nil, fmt.Errorf("elicit: nil params")
		}

		stream, hasStream := elicit.FromContext(ctx)
		if !hasStream {
			log.Printf("[elicit] no SSE stream in context — applying policy fallback for message=%q", truncate(params.Message, 80))
			return policyDecision(params), nil
		}

		cid, ch, unregister := pending.Register()
		defer unregister()

		log.Printf("[elicit] push input-required cid=%s msg=%q", cid, truncate(params.Message, 80))
		if err := stream.SendEvent(map[string]any{
			"status":         "input-required",
			"correlation_id": cid,
			"message":        params.Message,
			"schema":         params.RequestedSchema,
		}); err != nil {
			return nil, fmt.Errorf("elicit: push event: %w", err)
		}

		select {
		case res := <-ch:
			log.Printf("[elicit] peer responded cid=%s action=%s", cid, res.Action)
			return &mcp.ElicitResult{
				Action:  res.Action,
				Content: res.Content,
			}, nil
		case <-time.After(ElicitTimeout):
			log.Printf("[elicit] timeout waiting for peer reply cid=%s", cid)
			return &mcp.ElicitResult{Action: "cancel"}, nil
		case <-ctx.Done():
			log.Printf("[elicit] context cancelled cid=%s: %v", cid, ctx.Err())
			return &mcp.ElicitResult{Action: "cancel"}, ctx.Err()
		}
	}
}

// policyDecision returns the deterministic auto-decision used when no SSE
// stream is attached. Distinguishes bootstrap (single bool field) from
// duplicate (single enum string) by inspecting the schema shape.
func policyDecision(params *mcp.ElicitParams) *mcp.ElicitResult {
	// Bootstrap collection prompts have schema property "create" (bool).
	// Duplicate-action prompts have property "choice" (enum string).
	if schemaHasProperty(params.RequestedSchema, "create") {
		return &mcp.ElicitResult{
			Action:  "accept",
			Content: map[string]any{"create": true},
		}
	}
	// Default: decline. Includes duplicate-action where silent auto-accept
	// would risk data loss / churn.
	return &mcp.ElicitResult{Action: "decline"}
}

func schemaHasProperty(schema any, name string) bool {
	m, ok := schema.(map[string]any)
	if !ok {
		return false
	}
	props, ok := m["properties"].(map[string]any)
	if !ok {
		return false
	}
	_, has := props[name]
	return has
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
