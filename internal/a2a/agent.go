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
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"log"

	"github.com/modelcontextprotocol/go-sdk/mcp"
	"github.com/openai/openai-go"
	"github.com/openai/openai-go/shared"
	"github.com/pylyp-gh/ingest-orchestrator/internal/llm"
	"github.com/pylyp-gh/ingest-orchestrator/internal/mcpclient"
	"github.com/pylyp-gh/ingest-orchestrator/internal/peer"
	"github.com/pylyp-gh/ingest-orchestrator/internal/router"
	"github.com/pylyp-gh/ingest-orchestrator/internal/tracing"
	"go.opentelemetry.io/otel/attribute"
)

// maxIterations bounds the tool-use loop to prevent runaway behaviour
// (e.g., Claude calling the same tool in a cycle). Each iteration is one
// model round-trip + at most N tool calls. Bumped from 8 to 20 after a
// 3-block-plus-cross-reference team-coordinator task hit the ceiling
// при sequential delegation: each peer reply was a fresh iteration, plus
// follow-up clarification calls plus final synthesis. 20 gives headroom
// for a 5-peer team task with one round of clarifications each.
const maxIterations = 20

const systemPrompt = `You are the Ingest Orchestrator, an A2A team coordinator. You decompose user requests into subtasks and route them to either your own MCP tool (add_document) or to specialised kagent A2A peers via delegate_to_kagent_peer.

Your toolbelt:

1. add_document (MCP tool, local) — call doc-writer-mcp yourself. Use when the user provides direct text and you want fine-grained control (you see the action taken, point ID, metadata in the tool result).

2. delegate_to_kagent_peer (A2A) — invoke any kagent peer agent by name. See the tool description for the list of available peers and what each does. Use this to:
   - Hand off a single subtask to a specialist (e.g. k8s-agent for cluster queries)
   - Decompose a complex multi-domain request into parallel delegations (multiple tool_calls in one turn) and aggregate the results yourself

Anti-loop rule (critical, ENFORCED):

You may call each kagent peer (k8s-agent, helm-agent, doc-agent, etc.) AT MOST ONCE per turn. The orchestrator strictly enforces this: any second call to the same peer, no matter how you rephrase the prompt, returns the FIRST reply with a duplicate-call banner. There is no way to get new evidence from a peer mid-turn.

What this means for you:

- Bundle multiple questions into one delegation. If you need namespaces AND agent CRDs AND HTTPRoutes from k8s-agent, ask for all three in ONE delegate_to_kagent_peer call. Don't split into three calls hoping for cleaner replies, the second and third will be cached.

- Trust the first reply. If it is incomplete, surface that in your final answer ("doc-agent's reply did not cover X, so I cannot fully cross-reference") instead of trying a paraphrased re-query.

- Only retry on explicit error messages (the previous reply starts with "error:" or "error from peer:"). Cosmetic dissatisfaction is not grounds for retry.

For MCP tools (add_document, echo, etc.) the cache is keyed on the full arguments, so different parameters (different document content, different file path) trigger fresh calls as expected.

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
//
// If classifier is non-nil, a single Haiku classification call runs
// before the main loop. The resulting Verdict is attached to the
// per-turn context so every llm.Complete inside the loop honours the
// chosen tier (different baseURL + model). When classifier is nil
// (ROUTER_ENABLED=false or env unconfigured), Loop behaves exactly as
// before: Claude uses its configured primary tier with no per-turn
// override.
func Loop(ctx context.Context, claude *llm.Claude, mc *mcpclient.Client, discovery *peer.Discovery, classifier *router.Classifier, userText string) (answer string, err error) {
	ctx, rootSpan := tracing.Tracer().Start(ctx, "orchestrator.loop")
	tracing.SetKind(rootSpan, tracing.KindAgent)
	rootSpan.SetAttributes(
		attribute.Int("user.text_length", len(userText)),
		attribute.String("input.value", truncate(userText, 240)),
	)
	defer func() {
		rootSpan.SetAttributes(attribute.String("output.value", truncate(answer, 480)))
		tracing.EndWithErr(rootSpan, err)
	}()

	// Classify once per turn. Subsequent iterations of the tool-use loop
	// reuse the same Verdict via context, so the user pays for one
	// Haiku call per turn regardless of how many tool calls fire.
	if classifier != nil {
		verdict, cerr := classifier.Classify(ctx, userText)
		if cerr != nil {
			log.Printf("router: classify failed (%v), continuing with default tier", cerr)
		}
		ctx = router.WithVerdict(ctx, verdict)
		rootSpan.SetAttributes(
			attribute.String("router.tier", verdict.Tier.String()),
			attribute.String("router.reason", verdict.Reason),
		)
	}

	tools := mcpToolsToOpenAI(mc.Tools())
	var peers []peer.Peer
	if discovery != nil {
		peers = discovery.Peers()
	}
	tools = append(tools, peerToolDefinition(peers))
	rootSpan.SetAttributes(
		attribute.Int("tools.count", len(tools)),
		attribute.Int("peers.count", len(peers)),
	)

	messages := []openai.ChatCompletionMessageParamUnion{
		openai.SystemMessage(systemPrompt),
		openai.UserMessage(userText),
	}

	// Turn-scoped tool-call memoisation. Claude has been observed re-querying
	// the same peer with cosmetic prompt variations during cross-reference
	// tasks, burning iterations on redundant work. We hash (tool_name + args)
	// and short-circuit duplicates with the cached reply plus an inline note
	// so Claude sees that re-querying gave nothing new. Cache lives for the
	// duration of this Loop call only; next turn starts fresh.
	cache := map[string]string{}

	for i := 0; i < maxIterations; i++ {
		assistant, cerr := claude.Complete(ctx, messages, tools)
		if cerr != nil {
			err = fmt.Errorf("claude iteration %d: %w", i, cerr)
			return "", err
		}

		// Terminal text — no tool calls → return immediately.
		if len(assistant.ToolCalls) == 0 {
			rootSpan.SetAttributes(attribute.Int("loop.iterations", i+1))
			answer = assistant.Content
			return answer, nil
		}

		// Append assistant turn (with tool_calls) before sending tool results.
		messages = append(messages, assistant.ToParam())

		// Execute each tool call serially, append result messages.
		for _, call := range assistant.ToolCalls {
			result := executeToolCall(ctx, mc, call, cache)
			messages = append(messages, openai.ToolMessage(result, call.ID))
		}
	}

	err = fmt.Errorf("max iterations (%d) exceeded — Claude may be looping on tool calls", maxIterations)
	return "", err
}

// truncate clamps a string до n runes (NOT bytes) — byte-level slicing
// would split a multi-byte UTF-8 codepoint and crash OTLP export з
// "string field contains invalid UTF-8". Delegates to tracing.SafeTrunc
// so we have one source of truth у the codebase.
func truncate(s string, n int) string {
	return tracing.SafeTrunc(s, n)
}

// toolCacheKey derives a stable identifier for a tool call so duplicate
// invocations within the same turn can be short-circuited.
//
// For delegate_to_kagent_peer we key ONLY on agent_name and IGNORE the
// text argument. Observed behaviour: Claude paraphrases the same logical
// query into cosmetically different prompts ("comprehensive overview" /
// "detailed cluster state report" / etc.) hash that on raw args and the
// cache misses every time. Keying on agent alone enforces a strict
// one-call-per-peer-per-turn policy. Legitimate different sub-queries
// to the same peer (e.g. "list pods" then "describe pod foo") are
// collapsed too, but that is acceptable: the agent can answer multi-
// part questions in a single delegation, and the system prompt now
// instructs Claude to bundle questions accordingly.
//
// For other tools (MCP add_document, echo, anything that is not a peer
// delegation) we hash (tool_name + canonical args JSON): different
// args usually mean different real work.
func toolCacheKey(toolName, argsJSON string) string {
	if toolName == PeerToolName {
		// Extract agent_name and key only on that
		var args struct {
			AgentName string `json:"agent_name"`
		}
		_ = json.Unmarshal([]byte(argsJSON), &args)
		return "peer:" + args.AgentName
	}
	h := sha256.Sum256([]byte(toolName + "\x00" + argsJSON))
	return "tool:" + hex.EncodeToString(h[:12])
}

// executeToolCall invokes the named MCP tool with the JSON arguments from
// Claude. Returns a textual result string suitable for feeding back as a
// "tool" role message. Errors are surfaced as text so Claude can see them
// and adjust strategy (e.g., "duplicate exists — ask user").
//
// cache is a turn-scoped map of toolCacheKey -> previous result text.
// On cache hit we return the prior reply prepended with a banner so
// Claude sees the duplicate-call signal and (per system-prompt guidance)
// pivots to synthesis instead of re-issuing the same call again.
func executeToolCall(ctx context.Context, mc *mcpclient.Client, call openai.ChatCompletionMessageToolCall, cache map[string]string) string {
	name := call.Function.Name
	cacheKey := toolCacheKey(name, call.Function.Arguments)

	// Wrap entire tool dispatch у one span. Kind = AGENT для peer
	// delegation (downstream A2A invocation has its own loop), TOOL
	// otherwise. Set after we know the dispatch path below.
	ctx, span := tracing.Tracer().Start(ctx, "tool.call."+name)
	span.SetAttributes(
		attribute.String("tool.name", name),
		attribute.String("tool.cache_key", cacheKey),
		attribute.String("input.value", truncate(call.Function.Arguments, 480)),
	)
	defer span.End()

	// Memoisation: identical (tool, args) inside one Loop = duplicate.
	// Return prior reply with explicit duplicate-call notice so Claude's
	// next planning turn sees nothing new came back and pivots to
	// synthesising the answer from evidence already in the conversation.
	if prior, hit := cache[cacheKey]; hit {
		log.Printf("[agent] tool call: %s args=%s -> cached", name, call.Function.Arguments)
		span.SetAttributes(attribute.Bool("tool.cached", true))
		banner := "⚠️ duplicate call: same tool + arguments were already executed earlier in this turn. The previous reply is below; do NOT re-issue this call. Synthesise your final answer from the evidence already in your context.\n\n--- previous reply ---\n"
		out := banner + prior
		span.SetAttributes(attribute.String("output.value", truncate(out, 480)))
		return out
	}

	var args map[string]any
	if err := json.Unmarshal([]byte(call.Function.Arguments), &args); err != nil {
		tracing.SetKind(span, tracing.KindTool)
		msg := fmt.Sprintf("error: failed to parse tool arguments JSON: %v (raw: %q)", err, call.Function.Arguments)
		span.SetAttributes(attribute.String("output.value", msg))
		tracing.EndWithErr(span, err)
		// Errors NOT cached: a transient parse error shouldn't poison
		// the cache for a possible retry with corrected args.
		return msg
	}
	log.Printf("[agent] tool call: %s args=%s", name, call.Function.Arguments)

	// Dispatch — A2A peer delegation routes outside MCP entirely.
	if name == PeerToolName {
		tracing.SetKind(span, tracing.KindAgent)
		agentName, _ := args["agent_name"].(string)
		text, _ := args["text"].(string)
		span.SetAttributes(attribute.String("a2a.peer.name", agentName))
		if agentName == "" || text == "" {
			msg := "error: delegate_to_kagent_peer requires non-empty 'agent_name' and 'text' arguments"
			span.SetAttributes(attribute.String("output.value", msg))
			return msg
		}
		out, perr := invokeKagentPeer(ctx, agentName, text)
		if perr != nil {
			msg := fmt.Sprintf("error from peer %s: %v", agentName, perr)
			span.SetAttributes(attribute.String("output.value", msg))
			tracing.EndWithErr(span, perr)
			return msg
		}
		span.SetAttributes(attribute.String("output.value", truncate(out, 480)))
		cache[cacheKey] = out
		return out
	}

	tracing.SetKind(span, tracing.KindTool)
	resp, err := mc.CallTool(ctx, name, args)
	if err != nil {
		msg := fmt.Sprintf("error: tool call failed: %v", err)
		span.SetAttributes(attribute.String("output.value", msg))
		tracing.EndWithErr(span, err)
		return msg
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
	span.SetAttributes(
		attribute.Bool("tool.is_error", resp.IsError),
		attribute.String("output.value", truncate(out, 480)),
	)
	// Cache successful results only; tool errors (resp.IsError) are not
	// cached so Claude can retry with corrected inputs on the next pass.
	if !resp.IsError {
		cache[cacheKey] = out
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
