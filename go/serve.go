package main

// serve.go — HTTP chat server for nanollama
//
// Zero-dependency web UI. One binary, one command:
//   ./nanollama --model weights/model.gguf --serve --port 8080
//
// Endpoints:
//   GET  /        — chat UI (embedded HTML)
//   POST /chat    — generate text from JSON request
//   GET  /health  — model info

import (
	_ "embed"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"sync"
)

//go:embed ui.html
var uiHTML []byte

// chatRequest matches the frontend JSON format
type chatRequest struct {
	Messages    []chatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	MaxTokens   int           `json:"max_tokens"`
	TopK        int           `json:"top_k"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatResponse struct {
	Response string `json:"response"`
}

type healthResponse struct {
	Status    string `json:"status"`
	Params    int    `json:"params_millions"`
	Layers    int    `json:"layers"`
	Dim       int    `json:"dim"`
	Heads     int    `json:"heads"`
	KVHeads   int    `json:"kv_heads"`
	VocabSize int    `json:"vocab_size"`
	Gamma     bool   `json:"gamma_loaded"`
}

// runServer starts the HTTP chat server
func runServer(engine *Engine, defaults GenParams, port int) {
	// Mutex — model is single-threaded, one request at a time
	var mu sync.Mutex

	// GET / — serve chat UI
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/" {
			http.NotFound(w, r)
			return
		}
		w.Header().Set("Content-Type", "text/html; charset=utf-8")
		w.Write(uiHTML)
	})

	// POST /chat — generate text
	http.HandleFunc("/chat", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", http.StatusMethodNotAllowed)
			return
		}

		var req chatRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request: "+err.Error(), http.StatusBadRequest)
			return
		}

		if len(req.Messages) == 0 {
			writeJSON(w, chatResponse{Response: "Send a message."})
			return
		}

		// Build prompt from last user message
		prompt := req.Messages[len(req.Messages)-1].Content
		if prompt == "" {
			writeJSON(w, chatResponse{Response: "Empty message."})
			return
		}

		// Override defaults with request params
		params := defaults
		if req.MaxTokens > 0 {
			params.MaxTokens = req.MaxTokens
		}
		if req.Temperature > 0 {
			params.Temperature = float32(req.Temperature)
		}
		if req.TopK > 0 {
			params.TopK = req.TopK
		}

		// Generate (single-threaded, mutex protected)
		mu.Lock()
		result := engine.GenerateQuiet(prompt, params)
		mu.Unlock()

		writeJSON(w, chatResponse{Response: result})
	})

	// GET /health — model info
	http.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		cfg := engine.model.Config
		writeJSON(w, healthResponse{
			Status:    "ok",
			Params:    estimateParams(engine.model) / 1_000_000,
			Layers:    cfg.NumLayers,
			Dim:       cfg.EmbedDim,
			Heads:     cfg.NumHeads,
			KVHeads:   cfg.NumKVHeads,
			VocabSize: cfg.VocabSize,
			Gamma:     engine.model.Gamma != nil,
		})
	})

	addr := fmt.Sprintf(":%d", port)
	fmt.Printf("[nanollama] web UI at http://localhost:%d\n", port)
	log.Fatal(http.ListenAndServe(addr, nil))
}

func writeJSON(w http.ResponseWriter, v any) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(v)
}
