package router

import (
	"fmt"
	"strings"
)

// Verdict is the classifier's decision for a single user turn. Stored in
// the request context (see context.go) so that every downstream
// llm.Complete inside the loop picks up the same tier choice without
// re-classifying.
type Verdict struct {
	Tier   Tier
	Reason string // one-sentence justification, recorded as span attr
}

// ParseVerdict expects the classifier's reply to start with exactly one
// of TIER0/TIER1/TIER2/TIER3, optionally followed by a colon and a free-
// form reason. The first line is consumed; the rest is ignored so we
// tolerate a chatty Haiku that adds explanation lines after the verdict.
//
// On malformed input we still return a usable Verdict (Tier2 + raw text
// as Reason) plus a non-nil error. Callers should log the error but keep
// going so a single classifier hiccup doesn't break the request.
func ParseVerdict(s string) (Verdict, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return Verdict{Tier: Tier2, Reason: "(empty classifier reply)"},
			fmt.Errorf("router: classifier returned empty reply")
	}

	// First line, first whitespace-split token is the tier label. The
	// remainder of the first line (after the optional colon) is the reason.
	firstLine := s
	if idx := strings.IndexByte(s, '\n'); idx >= 0 {
		firstLine = s[:idx]
	}
	firstLine = strings.TrimSpace(firstLine)

	token := firstLine
	reason := ""
	if idx := strings.IndexByte(firstLine, ':'); idx >= 0 {
		token = strings.TrimSpace(firstLine[:idx])
		reason = strings.TrimSpace(firstLine[idx+1:])
	} else if idx := strings.IndexAny(firstLine, " \t"); idx >= 0 {
		token = strings.TrimSpace(firstLine[:idx])
		reason = strings.TrimSpace(firstLine[idx+1:])
	}

	tier, perr := ParseTier(token)
	if perr != nil {
		return Verdict{Tier: Tier2, Reason: firstLine}, perr
	}
	if reason == "" {
		reason = "(no reason supplied)"
	}
	return Verdict{Tier: tier, Reason: reason}, nil
}
