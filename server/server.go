// Package server exposes a forge Agent behind an OpenAI-compatible HTTP API.
//
// It implements the minimal surface needed to point an existing OpenAI client
// (set OPENAI_BASE_URL) at a forged agent: GET /v1/models lists the available
// agents as models, and POST /v1/chat/completions runs the named agent's full
// loop (the Engine owns the loop) and returns an OpenAI chat completion.
//
// The agent name is the OpenAI "model" field. The server is stateless: an
// OpenAI client sends the full message history on every call, so the agents it
// serves should be created with forge.Config{DisableMemory: true}.
package server

import (
	"bytes"
	"compress/flate"
	"compress/gzip"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/katasec/forge-core"
	"github.com/katasec/forge-core/message"
	"github.com/klauspost/compress/zstd"
)

// Server routes OpenAI-compatible requests to forge Agents keyed by model name.
type Server struct {
	agents       map[string]*forge.Agent
	defaultAgent string
	mux          *http.ServeMux
}

// New builds a Server that serves the given agents. The map key is the model
// name a client requests (e.g. "forged_reviewer").
//
// defaultAgent names the agent to use when a client requests a model that is
// not a known agent — host GUIs/CLIs send their own model ids (e.g. "gpt-5.5")
// that won't match an agent name. If defaultAgent is "" or unknown, an
// unrecognized model returns 404 (strict mode).
func New(agents map[string]*forge.Agent, defaultAgent string) *Server {
	s := &Server{
		agents:       agents,
		defaultAgent: defaultAgent,
		mux:          http.NewServeMux(),
	}
	s.mux.HandleFunc("GET /v1/models", s.handleModels)
	s.mux.HandleFunc("POST /v1/chat/completions", s.handleChatCompletions)
	s.mux.HandleFunc("POST /v1/responses", s.handleResponses)
	return s
}

// resolve maps a requested model name to an agent, falling back to the
// configured default agent for unrecognized model ids. It returns the agent and
// the resolved agent name, or ok=false when no agent applies.
func (s *Server) resolve(model string) (*forge.Agent, string, bool) {
	if a, ok := s.agents[model]; ok {
		return a, model, true
	}
	if s.defaultAgent != "" {
		if a, ok := s.agents[s.defaultAgent]; ok {
			return a, s.defaultAgent, true
		}
	}
	return nil, "", false
}

// ServeHTTP implements http.Handler. It logs each request's method, path, and
// status so the serve path is debuggable when a host CLI (e.g. Codex) points at
// it — an unexpected 404 on /v1/responses immediately shows a wire-API mismatch.
func (s *Server) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	sw := &statusWriter{ResponseWriter: w, status: http.StatusOK}
	s.mux.ServeHTTP(sw, r)
	log.Printf("%s %s -> %d", r.Method, r.URL.Path, sw.status)
}

// statusWriter captures the response status for logging while preserving the
// http.Flusher behaviour the streaming path depends on.
type statusWriter struct {
	http.ResponseWriter
	status int
}

func (w *statusWriter) WriteHeader(code int) {
	w.status = code
	w.ResponseWriter.WriteHeader(code)
}

func (w *statusWriter) Flush() {
	if f, ok := w.ResponseWriter.(http.Flusher); ok {
		f.Flush()
	}
}

// readBody reads and decompresses an HTTP request body. Host CLIs (notably
// Codex) send the request zstd- or gzip-compressed; it honours Content-Encoding
// and falls back to sniffing the magic bytes when the header is absent.
func readBody(r *http.Request) ([]byte, error) {
	raw, err := io.ReadAll(r.Body)
	if err != nil {
		return nil, err
	}

	enc := strings.ToLower(strings.TrimSpace(r.Header.Get("Content-Encoding")))
	if enc == "" {
		switch {
		case len(raw) >= 4 && raw[0] == 0x28 && raw[1] == 0xb5 && raw[2] == 0x2f && raw[3] == 0xfd:
			enc = "zstd"
		case len(raw) >= 2 && raw[0] == 0x1f && raw[1] == 0x8b:
			enc = "gzip"
		}
	}

	switch enc {
	case "zstd":
		zr, err := zstd.NewReader(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer zr.Close()
		return io.ReadAll(zr)
	case "gzip":
		gz, err := gzip.NewReader(bytes.NewReader(raw))
		if err != nil {
			return nil, err
		}
		defer gz.Close()
		return io.ReadAll(gz)
	case "deflate":
		fr := flate.NewReader(bytes.NewReader(raw))
		defer fr.Close()
		return io.ReadAll(fr)
	default:
		return raw, nil
	}
}

// --- OpenAI wire types (request) ---

type chatRequest struct {
	Model    string       `json:"model"`
	Messages []reqMessage `json:"messages"`
	Stream   bool         `json:"stream,omitempty"`
}

type reqMessage struct {
	Role    string       `json:"role"`
	Content contentField `json:"content"`
}

// contentField decodes message content that may be either a plain string or an
// array of typed parts. It covers both Chat Completions parts
// ([{type:"text",text:"..."}]) and Responses parts
// ([{type:"input_text"|"output_text",text:"..."}]) by concatenating every
// part's text regardless of part type.
type contentField struct {
	text string
}

func (c *contentField) UnmarshalJSON(b []byte) error {
	var s string
	if err := json.Unmarshal(b, &s); err == nil {
		c.text = s
		return nil
	}
	var parts []struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(b, &parts); err == nil {
		var sb strings.Builder
		for _, p := range parts {
			sb.WriteString(p.Text)
		}
		c.text = sb.String()
		return nil
	}
	// Tolerate unexpected shapes (objects, nulls, non-text parts) rather than
	// failing the whole request; best-effort text extraction only.
	return nil
}

// --- OpenAI wire types (response) ---

type chatResponse struct {
	ID      string       `json:"id"`
	Object  string       `json:"object"`
	Created int64        `json:"created"`
	Model   string       `json:"model"`
	Choices []respChoice `json:"choices"`
	Usage   usage        `json:"usage"`
}

type respChoice struct {
	Index        int         `json:"index"`
	Message      respMessage `json:"message"`
	FinishReason string      `json:"finish_reason"`
}

type respMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}

// --- streaming wire types ---

type streamChunk struct {
	ID      string         `json:"id"`
	Object  string         `json:"object"`
	Created int64          `json:"created"`
	Model   string         `json:"model"`
	Choices []streamChoice `json:"choices"`
}

type streamChoice struct {
	Index        int    `json:"index"`
	Delta        delta  `json:"delta"`
	FinishReason string `json:"finish_reason,omitempty"`
}

type delta struct {
	Role    string `json:"role,omitempty"`
	Content string `json:"content,omitempty"`
}

// --- models endpoint ---

type modelList struct {
	Object string      `json:"object"`
	Data   []modelInfo `json:"data"`
}

type modelInfo struct {
	ID      string `json:"id"`
	Object  string `json:"object"`
	Created int64  `json:"created"`
	OwnedBy string `json:"owned_by"`
}

func (s *Server) handleModels(w http.ResponseWriter, _ *http.Request) {
	now := time.Now().Unix()
	list := modelList{Object: "list"}
	for name := range s.agents {
		list.Data = append(list.Data, modelInfo{
			ID:      name,
			Object:  "model",
			Created: now,
			OwnedBy: "forge",
		})
	}
	writeJSON(w, http.StatusOK, list)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	data, err := readBody(r)
	if err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "could not read request body")
		return
	}
	var req chatRequest
	if err := json.Unmarshal(data, &req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid_request_error", "could not parse request body")
		return
	}

	log.Printf("chat: model=%q stream=%v messages=%d", req.Model, req.Stream, len(req.Messages))

	agent, resolved, ok := s.resolve(req.Model)
	if !ok {
		writeError(w, http.StatusNotFound, "invalid_request_error",
			fmt.Sprintf("model %q does not exist", req.Model))
		return
	}
	if resolved != req.Model {
		log.Printf("chat: model %q -> agent %q (default)", req.Model, resolved)
	}

	resp, err := agent.Run(r.Context(), forge.AgentRequest{
		Messages: translateMessages(req.Messages),
	})
	if err != nil {
		writeError(w, http.StatusBadGateway, "upstream_error", err.Error())
		return
	}

	id := "chatcmpl-" + uuid.NewString()
	created := time.Now().Unix()
	text := resp.LastText()
	finish := toOAIFinish(resp.FinishReason)

	if req.Stream {
		writeStream(w, id, created, req.Model, text, finish)
		return
	}

	writeJSON(w, http.StatusOK, chatResponse{
		ID:      id,
		Object:  "chat.completion",
		Created: created,
		Model:   req.Model,
		Choices: []respChoice{{
			Index:        0,
			Message:      respMessage{Role: "assistant", Content: text},
			FinishReason: finish,
		}},
		Usage: usage{
			PromptTokens:     resp.Usage.InputTokens,
			CompletionTokens: resp.Usage.OutputTokens,
			TotalTokens:      totalTokens(resp.Usage),
		},
	})
}

// translateMessages maps OpenAI messages to forge messages 1:1. The agent's own
// scaffold remains the authoritative system prompt; any client-supplied system
// message flows through as a forge system message.
func translateMessages(msgs []reqMessage) []forge.Message {
	out := make([]forge.Message, 0, len(msgs))
	for _, m := range msgs {
		out = append(out, forge.Message{
			Role:    toForgeRole(m.Role),
			Content: []forge.ContentBlock{message.Text(m.Content.text)},
		})
	}
	return out
}

func toForgeRole(r string) forge.Role {
	switch r {
	case "system":
		return forge.RoleSystem
	case "assistant":
		return forge.RoleAssistant
	case "tool":
		return forge.RoleTool
	default:
		return forge.RoleUser
	}
}

func toOAIFinish(r forge.FinishReason) string {
	switch r {
	case forge.FinishReasonIterLimit:
		return "length"
	default:
		// stop, error, and tool_use (already resolved by the loop) all present
		// as a completed turn to the client.
		return "stop"
	}
}

func totalTokens(u forge.TokenUsage) int {
	if u.TotalTokens > 0 {
		return u.TotalTokens
	}
	return u.InputTokens + u.OutputTokens
}

// writeStream emits the completed response as SSE chat.completion.chunk frames.
//
// The Engine runs the loop to completion before this is called, so the content
// is delivered in a single delta rather than token-by-token. The endpoint still
// speaks SSE so that clients requiring stream:true work; true token streaming
// depends on a streaming Provider, which the Engine does not yet expose.
func writeStream(w http.ResponseWriter, id string, created int64, model, text, finish string) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		writeError(w, http.StatusInternalServerError, "server_error", "streaming unsupported")
		return
	}

	h := w.Header()
	h.Set("Content-Type", "text/event-stream")
	h.Set("Cache-Control", "no-cache")
	h.Set("Connection", "keep-alive")
	w.WriteHeader(http.StatusOK)

	send := func(c streamChunk) {
		b, err := json.Marshal(c)
		if err != nil {
			log.Printf("chat stream: marshal chunk: %v", err)
			return
		}
		fmt.Fprintf(w, "data: %s\n\n", b)
		flusher.Flush()
	}

	base := func() streamChunk {
		return streamChunk{ID: id, Object: "chat.completion.chunk", Created: created, Model: model}
	}

	role := base()
	role.Choices = []streamChoice{{Index: 0, Delta: delta{Role: "assistant"}}}
	send(role)

	content := base()
	content.Choices = []streamChoice{{Index: 0, Delta: delta{Content: text}}}
	send(content)

	stop := base()
	stop.Choices = []streamChoice{{Index: 0, Delta: delta{}, FinishReason: finish}}
	send(stop)

	fmt.Fprint(w, "data: [DONE]\n\n")
	flusher.Flush()
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("writeJSON: encode response: %v", err)
	}
}

func writeError(w http.ResponseWriter, status int, errType, msg string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    errType,
		},
	})
}
