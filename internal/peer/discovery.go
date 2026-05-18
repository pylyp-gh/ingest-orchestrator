// Package peer implements auto-discovery of kagent A2A peers in the
// cluster. Replaces the hardcoded peer list with a live view derived
// from the kagent Agent CRDs running at the time of each agent loop.
//
// Mechanism:
//
//  1. List Agent CRs (`agents.kagent.dev/v1alpha2`) in the kagent ns
//     via direct HTTP to the Kubernetes API server. No client-go
//     dependency — uses the pod's projected ServiceAccount token +
//     mounted CA cert to authenticate. ~80 lines vs ~30MB of client-go.
//
//  2. For each Agent CR, fetch its native A2A Agent Card at
//     <name>.<ns>.svc.cluster.local:8080/.well-known/agent-card.json.
//     The card's `description` field is the authoritative source for
//     the peer's capabilities — same string a human sees when curling
//     the well-known URI.
//
//  3. Build a []Peer slice atomically stored in an atomic.Pointer for
//     lock-free reads by the agent Loop's tool-definition builder.
//
// Refresh: synchronous initial fetch at startup (Loop fails clear if
// the cluster has no peers reachable); periodic refresh via ticker
// running until ctx.Done.
//
// RBAC: requires ClusterRole with get,list on agents.kagent.dev. See
// clusters/kind-lab/apps/ingest-orchestrator/controller/rbac.yaml.
package peer

import (
	"context"
	"crypto/tls"
	"crypto/x509"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"sync/atomic"
	"time"
)

const (
	saTokenPath = "/var/run/secrets/kubernetes.io/serviceaccount/token"
	saCAPath    = "/var/run/secrets/kubernetes.io/serviceaccount/ca.crt"
	apiServer   = "https://kubernetes.default.svc"
	agentsAPI   = "/apis/kagent.dev/v1alpha2/namespaces/%s/agents"
)

// Peer is one kagent A2A agent discoverable in the cluster.
type Peer struct {
	Name        string
	Description string
	Endpoint    string
}

// Discovery polls the kagent ns for Agent CRDs and their Agent Cards.
type Discovery struct {
	k8sClient *http.Client
	cardClient *http.Client
	token     string
	namespace string
	cache     atomic.Pointer[[]Peer]
}

// NewDiscovery loads the pod's ServiceAccount token + CA cert and
// constructs an authenticated HTTP client against the K8s API.
// Returns an error if the pod is not running inside a cluster (no SA
// token mounted) — caller can fall back to a static list if desired.
func NewDiscovery(namespace string) (*Discovery, error) {
	tokenBytes, err := os.ReadFile(saTokenPath)
	if err != nil {
		return nil, fmt.Errorf("read SA token: %w", err)
	}
	caBytes, err := os.ReadFile(saCAPath)
	if err != nil {
		return nil, fmt.Errorf("read SA CA cert: %w", err)
	}
	pool := x509.NewCertPool()
	if !pool.AppendCertsFromPEM(caBytes) {
		return nil, fmt.Errorf("SA CA cert: no PEM blocks parsed")
	}
	k8sClient := &http.Client{
		Timeout: 10 * time.Second,
		Transport: &http.Transport{
			TLSClientConfig: &tls.Config{RootCAs: pool},
		},
	}
	// Agent Cards served by kagent agents are plain HTTP on Service:8080.
	cardClient := &http.Client{Timeout: 5 * time.Second}
	return &Discovery{
		k8sClient:  k8sClient,
		cardClient: cardClient,
		token:      string(tokenBytes),
		namespace:  namespace,
	}, nil
}

// Refresh pulls the current Agent list from the K8s API, fetches each
// one's Agent Card, and atomically replaces the cached peer slice.
// Agents whose card fetch fails are skipped with a warning (don't
// block the whole refresh on one bad agent).
func (d *Discovery) Refresh(ctx context.Context) error {
	url := apiServer + fmt.Sprintf(agentsAPI, d.namespace)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return fmt.Errorf("build request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+d.token)
	req.Header.Set("Accept", "application/json")

	resp, err := d.k8sClient.Do(req)
	if err != nil {
		return fmt.Errorf("list agents: %w", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return fmt.Errorf("list agents HTTP %d: %s", resp.StatusCode, truncate(string(body), 200))
	}

	var list struct {
		Items []struct {
			Metadata struct {
				Name      string `json:"name"`
				Namespace string `json:"namespace"`
			} `json:"metadata"`
		} `json:"items"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&list); err != nil {
		return fmt.Errorf("decode agent list: %w", err)
	}

	peers := make([]Peer, 0, len(list.Items))
	for _, item := range list.Items {
		card, err := d.fetchAgentCard(ctx, item.Metadata.Name, item.Metadata.Namespace)
		if err != nil {
			log.Printf("[discovery] skip %s/%s — card fetch failed: %v", item.Metadata.Namespace, item.Metadata.Name, err)
			continue
		}
		peers = append(peers, Peer{
			Name:        item.Metadata.Name,
			Description: card.Description,
			Endpoint:    fmt.Sprintf("http://%s.%s.svc.cluster.local:8080/", item.Metadata.Name, item.Metadata.Namespace),
		})
	}
	d.cache.Store(&peers)
	log.Printf("[discovery] refreshed: %d peers found (ns=%s)", len(peers), d.namespace)
	return nil
}

// Peers returns the current cached peer slice. Safe for concurrent
// readers from any goroutine. Returns nil if Refresh has never run.
func (d *Discovery) Peers() []Peer {
	p := d.cache.Load()
	if p == nil {
		return nil
	}
	return *p
}

// Run launches a periodic refresh ticker; blocks until ctx cancelled.
func (d *Discovery) Run(ctx context.Context, interval time.Duration) {
	t := time.NewTicker(interval)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-t.C:
			if err := d.Refresh(ctx); err != nil {
				log.Printf("[discovery] periodic refresh failed: %v", err)
			}
		}
	}
}

type agentCard struct {
	Description string `json:"description"`
}

func (d *Discovery) fetchAgentCard(ctx context.Context, name, ns string) (*agentCard, error) {
	url := fmt.Sprintf("http://%s.%s.svc.cluster.local:8080/.well-known/agent-card.json", name, ns)
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	resp, err := d.cardClient.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	var card agentCard
	if err := json.NewDecoder(resp.Body).Decode(&card); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}
	return &card, nil
}

func truncate(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n] + "…"
}
