// Package mcpclient wraps the modelcontextprotocol/go-sdk client for use by
// the agent loop. Establishes Streamable HTTP sessions to a target MCP
// server (doc-writer-mcp), exposes ListTools and CallTool.
//
// Phase 3 (Sampling): when the MCP server issues a sampling/createMessage
// request, the CreateMessageHandler translates SamplingMessages into an
// OpenAI ChatCompletion call against agentgateway and returns the
// completion.
//
// Phase 4 (Elicitation): when the server issues elicitation/create, the
// handler may either bridge to an SSE peer ("interactive" handler used
// for /messages:stream requests) or apply a deterministic policy
// fallback ("policy" handler used for sync /messages requests).
//
// Two constructors:
//
//   - New(): shared session used by the sync /messages handler. Policy
//     ElicitationHandler — no peer to ask, so auto-decisions per schema.
//
//   - NewWithStream(): per-request session created for each streaming
//     request. ElicitationHandler closure captures the request's SSE
//     stream, fully isolating concurrent streams from each other.
//
// Why per-request session for streaming: MCP SDK invokes Elicitation /
// Sampling handlers from an async dispatcher goroutine whose ctx is
// transport-level — not the ctx the caller passed to CallTool. The only
// way to get a per-request stream visible to the handler is to close it
// in at session construction. Connection overhead (initialize + ListTools)
// is paid per stream — acceptable for interactive use, optimisable later.
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

// ElicitTimeout bounds how long the interactive elicit bridge waits for
// a peer reply before falling back to "cancel". Generous so a human peer
// can reason about the prompt, but short enough that orphaned MCP calls
// don't pile up if peer disconnected.
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

func mcpServerURL() string {
	return envOr("MCP_SERVER_URL", "https://doc-writer.ash.ph.lab/mcp")
}

func implementation() *mcp.Implementation {
	return &mcp.Implementation{
		Name:    "ingest-orchestrator",
		Version: "0.1.0",
	}
}

// New connects a shared, long-lived MCP client used by the sync /messages
// endpoint. ElicitationHandler uses the policy fallback (no peer stream
// to push prompts at). Sampling handler bridges to Claude via gateway.
func New(ctx context.Context, claude *llm.Claude) (*Client, error) {
	opts := &mcp.ClientOptions{
		CreateMessageHandler: buildSamplingHandler(claude),
		ElicitationHandler:   buildPolicyElicitHandler(),
	}
	return connect(ctx, opts)
}

// NewWithStream connects a per-request MCP client whose ElicitationHandler
// closure captures the given SSE stream. Each streaming request creates
// one of these, uses it for the agent loop, then Close()s it. Concurrent
// streams get independent sessions — no shared state, no head-of-line
// blocking.
//
// pending is the process-wide PendingRegistry — correlation IDs are
// globally unique, so multiple per-request sessions safely share it.
func NewWithStream(ctx context.Context, claude *llm.Claude, pending *elicit.PendingRegistry, stream *elicit.Stream) (*Client, error) {
	opts := &mcp.ClientOptions{
		CreateMessageHandler: buildSamplingHandler(claude),
		ElicitationHandler:   buildInteractiveElicitHandler(pending, stream),
	}
	return connect(ctx, opts)
}

// connect shares the boilerplate between New / NewWithStream.
func connect(ctx context.Context, opts *mcp.ClientOptions) (*Client, error) {
	url := mcpServerURL()
	mcpClient := mcp.NewClient(implementation(), opts)
	transport := &mcp.StreamableClientTransport{Endpoint: url}
	session, err := mcpClient.Connect(ctx, transport, nil)
	if err != nil {
		return nil, fmt.Errorf("mcp connect %s: %w", url, err)
	}
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
//   - params.Messages[].Content → only *TextContent supported
//
// Tools deliberately omitted: doc-writer-mcp's L5 calls are single-turn
// classification/extraction — no tool use needed.
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
				return nil, fmt.Errorf("sampling: message %d content type %T unsupported (text only)", i, m.Content)
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

// buildInteractiveElicitHandler returns an ElicitationHandler that pushes
// "input-required" events to the captured stream and blocks until the
// peer responds via /messages:respond with the matching correlation ID.
//
// The stream is captured by closure — each call to this builder produces
// a handler bound to one specific stream, used by exactly one
// /messages:stream request. Concurrent streams get independent handlers.
func buildInteractiveElicitHandler(pending *elicit.PendingRegistry, stream *elicit.Stream) func(context.Context, *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
	return func(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
		params := req.Params
		if params == nil {
			return nil, fmt.Errorf("elicit: nil params")
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

// buildPolicyElicitHandler returns an ElicitationHandler that applies a
// deterministic policy without consulting any peer. Used by the shared
// session (sync /messages endpoint) where there is nobody to ask.
//
// Policy:
//   - bootstrap (single bool field "create")  → accept create=true
//   - duplicate (single enum field "choice")  → decline
//   - any other schema                          → decline
//
// Bootstrap is safe to auto-accept: creating an empty Qdrant collection
// has no content risk. Duplicate auto-decline avoids silent data loss —
// caller (Claude agent) sees the decline and can re-ask the user.
func buildPolicyElicitHandler() func(context.Context, *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
	return func(ctx context.Context, req *mcp.ElicitRequest) (*mcp.ElicitResult, error) {
		params := req.Params
		if params == nil {
			return nil, fmt.Errorf("elicit: nil params")
		}
		log.Printf("[elicit:policy] msg=%q", truncate(params.Message, 80))
		return policyDecision(params), nil
	}
}

func policyDecision(params *mcp.ElicitParams) *mcp.ElicitResult {
	if schemaHasProperty(params.RequestedSchema, "create") {
		return &mcp.ElicitResult{
			Action:  "accept",
			Content: map[string]any{"create": true},
		}
	}
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
