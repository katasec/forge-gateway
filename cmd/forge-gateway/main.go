// Command forge-gateway serves forged agents behind a local OpenAI-compatible
// endpoint. It is the runnable proof of the serve path: it wires demo agents
// (defined in Go) to the OpenAI gateway so you can point any OpenAI client at
// it.
//
// Usage:
//
//	export OPENAI_API_KEY=sk-...            # key for the upstream provider
//	go run ./cmd/forge-gateway --addr :8787
//
//	export OPENAI_BASE_URL=http://localhost:8787/v1
//	export OPENAI_API_KEY=forge-local       # the gateway ignores this
//	# now start Codex CLI / Grok Build, or just curl /v1/chat/completions
//
// The two demo agents share one provider and model and differ only in their
// scaffold — the north-star demo: same model, same task, different package.
package main

import (
	"flag"
	"log"
	"net/http"
	"os"

	"github.com/katasec/forge-core"
	"github.com/katasec/forge-core/provider/openai"
	"github.com/katasec/forge-gateway/server"
)

const forgedReviewerScaffold = `You are a repository reviewer operating under a mission scaffold.

Operating rules:
- Start from a small orientation layer; do not assume the whole repo.
- Ground every finding in a concrete file, command, or repo fact — no generic advice.
- Identify exactly one concrete, high-value next improvement.
- Recommend the verification (build/test/lint) that proves the change.
- Keep output structured: Findings, Next Improvement, Verification.`

const vanillaScaffold = "You are a helpful coding assistant. Review the repository."

func main() {
	addr := flag.String("addr", ":8787", "address to listen on")
	model := flag.String("model", string(openai.ModelGPT54Nano), "upstream model id")
	baseURL := flag.String("base-url", "", "override upstream OpenAI base URL (e.g. for xAI)")
	defaultAgent := flag.String("default-agent", "forged_reviewer",
		"agent to use when a client requests an unknown model id (host GUIs send their own model names); empty for strict 404")
	flag.Parse()

	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		log.Fatal("OPENAI_API_KEY is required (the upstream provider key)")
	}

	var opts []openai.Option
	if *baseURL != "" {
		opts = append(opts, openai.WithBaseURL(*baseURL))
	}
	provider := openai.New(apiKey, openai.Model(*model), opts...)

	agents := map[string]*forge.Agent{
		"vanilla_reviewer": mustAgent(provider, vanillaScaffold),
		"forged_reviewer":  mustAgent(provider, forgedReviewerScaffold),
	}

	srv := server.New(agents, *defaultAgent)
	log.Printf("forge-gateway serving %d agents on %s (upstream model %s, default agent %q)", len(agents), *addr, *model, *defaultAgent)
	log.Printf("point your client at: export OPENAI_BASE_URL=http://localhost%s/v1", *addr)
	if err := http.ListenAndServe(*addr, srv); err != nil {
		log.Fatal(err)
	}
}

func mustAgent(provider forge.Provider, scaffold string) *forge.Agent {
	agent, err := forge.NewAgent(forge.Config{
		Provider:      provider,
		SystemPrompt:  scaffold,
		DisableMemory: true, // OpenAI clients are stateless; they resend history
	})
	if err != nil {
		log.Fatal(err)
	}
	return agent
}
