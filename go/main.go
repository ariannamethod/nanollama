package main

// main.go — nanollama Go inference engine
//
// Zero-dependency LLaMA inference from GGUF files.
// Supports personality injection via γ (gamma) — θ = ε + γ
//
// Usage:
//   go build -o nanollama .
//   ./nanollama --model weights/model.gguf --prompt "Hello world"
//   ./nanollama --model weights/model.gguf --gamma weights/gamma.npz --interactive

import (
	"bufio"
	"flag"
	"fmt"
	"math"
	"math/rand"
	"os"
	"sort"
	"strings"
	"time"
)

func main() {
	modelPath := flag.String("model", "", "Path to GGUF model file (required)")
	gammaPath := flag.String("gamma", "", "Path to gamma NPZ file (optional, personality)")
	prompt := flag.String("prompt", "", "Prompt text (if empty, reads from stdin)")
	maxTokens := flag.Int("max-tokens", 256, "Maximum tokens to generate")
	temperature := flag.Float64("temp", 0.8, "Sampling temperature (0 = greedy)")
	topP := flag.Float64("top-p", 0.9, "Top-p (nucleus) sampling threshold")
	topK := flag.Int("top-k", 50, "Top-k sampling (used when top-p >= 1.0)")
	repPenalty := flag.Float64("rep-penalty", 1.15, "Repetition penalty (1.0 = disabled)")
	repWindow := flag.Int("rep-window", 64, "Repetition penalty lookback window")
	interactive := flag.Bool("interactive", false, "Interactive REPL mode")
	listTensors := flag.Bool("list-tensors", false, "List all tensors and exit")
	flag.Parse()

	if *modelPath == "" {
		fmt.Fprintln(os.Stderr, "Usage: nanollama --model <path.gguf> [--gamma <path.npz>] [--prompt <text>]")
		fmt.Fprintln(os.Stderr, "")
		fmt.Fprintln(os.Stderr, "Flags:")
		flag.PrintDefaults()
		os.Exit(1)
	}

	// Load GGUF
	fmt.Printf("[nanollama] loading %s\n", *modelPath)
	gguf, err := LoadGGUF(*modelPath)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	if *listTensors {
		gguf.ListTensors()
		return
	}

	// Load model
	model, err := LoadLlamaModel(gguf)
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Load gamma (personality)
	if *gammaPath != "" {
		gamma, err := LoadGamma(*gammaPath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "warning: failed to load gamma: %v\n", err)
		} else {
			if gamma.EmbedDim != model.Config.EmbedDim {
				fmt.Fprintf(os.Stderr, "warning: gamma embed_dim %d != model dim %d, skipping\n",
					gamma.EmbedDim, model.Config.EmbedDim)
			} else {
				model.Gamma = gamma
				fmt.Printf("[nanollama] personality loaded: %d tokens modified\n", gamma.NumTokens)
			}
		}
	}

	// Build tokenizer
	tokenizer := NewTokenizer(&gguf.Meta)

	// Create engine
	engine := &Engine{
		model:      model,
		tokenizer:  tokenizer,
		rng:        rand.New(rand.NewSource(time.Now().UnixNano())),
		repPenalty: float32(*repPenalty),
		repWindow:  *repWindow,
	}

	params := GenParams{
		MaxTokens:   *maxTokens,
		Temperature: float32(*temperature),
		TopP:        float32(*topP),
		TopK:        *topK,
	}

	fmt.Printf("[nanollama] ready — %dM params, %d layers, %d dim\n",
		estimateParams(model)/1_000_000, model.Config.NumLayers, model.Config.EmbedDim)

	if *interactive {
		runREPL(engine, params)
	} else if *prompt != "" {
		result := engine.Generate(*prompt, params)
		fmt.Println(result)
	} else {
		// Read from stdin
		scanner := bufio.NewScanner(os.Stdin)
		fmt.Print("> ")
		for scanner.Scan() {
			text := strings.TrimSpace(scanner.Text())
			if text == "" {
				fmt.Print("> ")
				continue
			}
			if text == "/quit" || text == "/exit" {
				break
			}
			result := engine.Generate(text, params)
			fmt.Println(result)
			fmt.Print("\n> ")
		}
	}
}

// GenParams controls text generation
type GenParams struct {
	MaxTokens   int
	Temperature float32
	TopP        float32
	TopK        int
}

// Engine wraps model + tokenizer for generation
type Engine struct {
	model      *LlamaModel
	tokenizer  *Tokenizer
	rng        *rand.Rand
	repPenalty float32
	repWindow  int
}

// Generate produces text from a prompt
func (e *Engine) Generate(prompt string, p GenParams) string {
	// Tokenize with BOS
	tokens := e.tokenizer.Encode(prompt, true)

	e.model.Reset()

	// Prefill: feed all prompt tokens
	pos := 0
	for _, tok := range tokens {
		e.model.Forward(tok, pos)
		pos++
		if pos >= e.model.Config.SeqLen-1 {
			break
		}
	}

	// Generate
	var output []byte
	recentTokens := make([]int, 0, e.repWindow)
	startTime := time.Now()

	for i := 0; i < p.MaxTokens && len(output) < 8192; i++ {
		logits := e.model.State.Logits

		// Repetition penalty
		if e.repPenalty > 1.0 && len(recentTokens) > 0 {
			for _, tok := range recentTokens {
				if tok >= 0 && tok < e.model.Config.VocabSize {
					if logits[tok] > 0 {
						logits[tok] /= e.repPenalty
					} else {
						logits[tok] *= e.repPenalty
					}
				}
			}
		}

		// Sample
		var next int
		if p.TopP < 1.0 {
			next = e.sampleTopP(p.Temperature, p.TopP)
		} else {
			next = e.sampleTopK(p.Temperature, p.TopK)
		}

		recentTokens = append(recentTokens, next)
		if len(recentTokens) > e.repWindow {
			recentTokens = recentTokens[1:]
		}

		// Stop on EOS
		if next == e.tokenizer.EosID {
			break
		}

		piece := e.tokenizer.DecodeToken(next)
		output = append(output, []byte(piece)...)

		// Stream output
		fmt.Print(piece)

		e.model.Forward(next, pos)
		pos++

		if pos >= e.model.Config.SeqLen {
			break
		}
	}
	fmt.Println()

	elapsed := time.Since(startTime)
	tokensGen := len(recentTokens)
	if elapsed.Seconds() > 0 && tokensGen > 0 {
		tps := float64(tokensGen) / elapsed.Seconds()
		fmt.Printf("[%d tokens, %.1f tok/s]\n", tokensGen, tps)
	}

	return string(output)
}

// sampleTopK samples from top-k logits
func (e *Engine) sampleTopK(temp float32, topK int) int {
	logits := e.model.State.Logits
	vocab := e.model.Config.VocabSize

	if temp <= 0 {
		return argmax(logits, vocab)
	}
	if topK > vocab {
		topK = vocab
	}

	type idxVal struct {
		idx int
		val float32
	}
	top := make([]idxVal, topK)
	for i := 0; i < topK; i++ {
		top[i] = idxVal{-1, -1e30}
	}

	for i := 0; i < vocab; i++ {
		if logits[i] > top[topK-1].val {
			top[topK-1] = idxVal{i, logits[i]}
			for j := topK - 1; j > 0 && top[j].val > top[j-1].val; j-- {
				top[j], top[j-1] = top[j-1], top[j]
			}
		}
	}

	maxVal := top[0].val
	probs := make([]float32, topK)
	var sum float32
	for i := 0; i < topK; i++ {
		if top[i].idx < 0 {
			break
		}
		probs[i] = float32(math.Exp(float64((top[i].val - maxVal) / temp)))
		sum += probs[i]
	}

	r := e.rng.Float32() * sum
	var cdf float32
	for i := 0; i < topK; i++ {
		cdf += probs[i]
		if r <= cdf {
			return top[i].idx
		}
	}
	return top[0].idx
}

// sampleTopP samples using nucleus (top-p) sampling
func (e *Engine) sampleTopP(temp, topP float32) int {
	logits := e.model.State.Logits
	vocab := e.model.Config.VocabSize

	if temp <= 0 {
		return argmax(logits, vocab)
	}

	maxVal := logits[0]
	for i := 1; i < vocab; i++ {
		if logits[i] > maxVal {
			maxVal = logits[i]
		}
	}

	type idxProb struct {
		idx  int
		prob float32
	}
	candidates := make([]idxProb, vocab)
	var sum float32
	for i := 0; i < vocab; i++ {
		p := float32(math.Exp(float64((logits[i] - maxVal) / temp)))
		candidates[i] = idxProb{i, p}
		sum += p
	}

	invSum := float32(1.0) / sum
	for i := range candidates {
		candidates[i].prob *= invSum
	}

	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	var cumsum float32
	for i := range candidates {
		cumsum += candidates[i].prob
		if cumsum >= topP {
			r := e.rng.Float32() * cumsum
			var cdf float32
			for j := 0; j <= i; j++ {
				cdf += candidates[j].prob
				if r <= cdf {
					return candidates[j].idx
				}
			}
			return candidates[0].idx
		}
	}
	return candidates[0].idx
}

func argmax(logits []float32, n int) int {
	best := 0
	for i := 1; i < n; i++ {
		if logits[i] > logits[best] {
			best = i
		}
	}
	return best
}

// estimateParams rough estimate of total parameters
func estimateParams(m *LlamaModel) int {
	cfg := &m.Config
	// embed + output + layers*(attn + mlp + norms)
	embed := cfg.VocabSize * cfg.EmbedDim
	output := cfg.VocabSize * cfg.EmbedDim
	attnPerLayer := cfg.EmbedDim*(cfg.NumHeads*cfg.HeadDim) + // Q
		cfg.EmbedDim*(cfg.NumKVHeads*cfg.HeadDim) + // K
		cfg.EmbedDim*(cfg.NumKVHeads*cfg.HeadDim) + // V
		cfg.EmbedDim*cfg.EmbedDim // O
	mlpPerLayer := cfg.EmbedDim*cfg.IntermSize*2 + // gate + up
		cfg.IntermSize*cfg.EmbedDim // down
	normsPerLayer := cfg.EmbedDim * 2
	perLayer := attnPerLayer + mlpPerLayer + normsPerLayer
	return embed + output + cfg.NumLayers*perLayer + cfg.EmbedDim
}

// runREPL starts an interactive session
func runREPL(engine *Engine, params GenParams) {
	scanner := bufio.NewScanner(os.Stdin)
	fmt.Println("nanollama interactive mode. Type /quit to exit.")
	fmt.Println()

	for {
		fmt.Print("> ")
		if !scanner.Scan() {
			break
		}
		text := strings.TrimSpace(scanner.Text())
		if text == "" {
			continue
		}

		switch text {
		case "/quit", "/exit":
			return
		case "/info":
			cfg := engine.model.Config
			fmt.Printf("Model: %d layers, %d dim, %d heads, %d kv_heads, %d vocab\n",
				cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize)
			fmt.Printf("Params: ~%dM\n", estimateParams(engine.model)/1_000_000)
			fmt.Printf("Gamma: %v\n", engine.model.Gamma != nil)
			fmt.Printf("Flags: qk_norm=%v rope_conjugate=%v\n", cfg.QKNorm, cfg.RopeConjugate)
			continue
		}

		engine.Generate(text, params)
		fmt.Println()
	}
}
