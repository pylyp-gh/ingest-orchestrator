package llm

import "errors"

var ErrNoChoices = errors.New("openai returned zero choices in completion response")
