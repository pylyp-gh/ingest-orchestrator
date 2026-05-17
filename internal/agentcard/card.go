// Package agentcard serves the A2A Agent Card at /.well-known/agent-card.json
// per RFC 8615 / A2A Protocol specification (a2a-protocol.org).
//
// The Agent Card describes who this agent is, what skills it offers, how to
// reach it, and which auth schemes it requires. External A2A peers discover
// the agent by GETting this URL with no auth and parsing the JSON document.
package agentcard

import (
	"encoding/json"
	"net/http"
	"os"
)

// Card mirrors the A2A AgentCard schema. Field names match the spec JSON
// directly so external A2A clients can parse without translation.
type Card struct {
	ID              string         `json:"id"`
	Name            string         `json:"name"`
	Description     string         `json:"description"`
	Provider        Provider       `json:"provider"`
	ServiceEndpoint string         `json:"serviceEndpoint"`
	Capabilities    Capabilities   `json:"capabilities"`
	Skills          []Skill        `json:"skills"`
	SecuritySchemes map[string]any `json:"securitySchemes,omitempty"`
	Security        []any          `json:"security,omitempty"`
}

type Provider struct {
	Organization string `json:"organization"`
	URL          string `json:"url,omitempty"`
}

type Capabilities struct {
	Streaming         bool `json:"streaming"`
	PushNotifications bool `json:"pushNotifications"`
	ExtendedAgentCard bool `json:"extendedAgentCard,omitempty"`
}

type Skill struct {
	ID          string   `json:"id"`
	Name        string   `json:"name"`
	Description string   `json:"description"`
	Examples    []string `json:"examples,omitempty"`
	Tags        []string `json:"tags,omitempty"`
}

func envOr(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// Default returns the Agent Card for this orchestrator. ServiceEndpoint is
// env-overridable so the same binary works from local dev (localhost),
// in-cluster (Service DNS), and through the gateway (public hostname).
func Default() Card {
	endpoint := envOr("A2A_SERVICE_ENDPOINT", "https://ingest.ash.ph.lab")
	return Card{
		ID:              "ingest-orchestrator",
		Name:            "Ingest Orchestrator",
		Description:     "A2A agent that orchestrates documentation ingest into Qdrant via the doc-writer-mcp tool. Bridges MCP server capabilities (Sampling, Elicitation) to external A2A peers, demonstrating a full-capability MCP client integrated with the Anthropic Claude API.",
		Provider:        Provider{Organization: "pylyp-gh", URL: "https://github.com/pylyp-gh"},
		ServiceEndpoint: endpoint,
		Capabilities:    Capabilities{Streaming: true, PushNotifications: false},
		Skills: []Skill{
			{
				ID:          "echo",
				Name:        "Echo",
				Description: "Returns the input text verbatim. Used for connectivity validation and A2A protocol smoke tests.",
				Examples:    []string{"echo hello world"},
				Tags:        []string{"diagnostic", "smoke-test"},
			},
		},
	}
}

// Handler returns an http.HandlerFunc that serves the well-known Agent Card
// path. GET only; no auth required (per A2A spec — Agent Card is the
// authentication-discovery document itself, so it must be openly reachable).
func Handler() http.HandlerFunc {
	card := Default()
	body, _ := json.MarshalIndent(card, "", "  ")
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodGet {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		w.Header().Set("Cache-Control", "public, max-age=60")
		w.Write(body)
	}
}
