package main

// model.go — LLaMA-family forward pass
//
// Llama 3 architecture with nanollama extensions:
//   - Parameterless RMSNorm (identity weights injected in GGUF)
//   - QK-norm: RMSNorm on Q,K after RoPE (stabilizes training)
//   - Conjugate RoPE: (x0*cos+x1*sin, -x0*sin+x1*cos)
//   - GQA (grouped query attention)
//   - SwiGLU MLP
//   - No bias on attention projections

import (
	"fmt"
	"math"
)

// LlamaModel is a loaded Llama model ready for inference
type LlamaModel struct {
	Config  LlamaConfig
	Weights LlamaWeights
	State   LlamaState
	Gamma   *GammaEssence // personality essence (nil = no gamma)
}

// LlamaConfig holds model dimensions
type LlamaConfig struct {
	NumLayers  int
	EmbedDim   int
	NumHeads   int
	NumKVHeads int
	HeadDim    int
	VocabSize  int
	SeqLen     int
	IntermSize int     // MLP intermediate dimension
	RMSNormEps float32
	RopeTheta  float32

	// nanollama-specific flags (read from GGUF metadata)
	QKNorm        bool // normalize Q,K with RMSNorm after RoPE
	RopeConjugate bool // conjugate RoPE: (x0*cos + x1*sin, -x0*sin + x1*cos)
}

// LlamaWeights holds all weight tensors (quantized raw bytes or F32 slices)
type LlamaWeights struct {
	// Token embedding [vocab, dim]
	TokenEmbed   []byte
	TokenEmbType uint32

	// Output norm [dim]
	OutputNorm []float32

	// Output (LM head) [vocab, dim]
	Output     []byte
	OutputType uint32

	// Per-layer weights
	Layers []LlamaLayerWeights
}

// LlamaLayerWeights holds weights for one transformer layer
type LlamaLayerWeights struct {
	// Attention norms
	AttnNorm []float32 // [dim]
	FFNNorm  []float32 // [dim]

	// Attention projections [out_dim, in_dim]
	WQ     []byte
	WQType uint32
	WK     []byte
	WKType uint32
	WV     []byte
	WVType uint32
	WO     []byte
	WOType uint32

	// Attention biases (optional — Qwen2.5 has these, nanollama does not)
	BQ []float32 // [num_heads * head_dim] — nil if no bias
	BK []float32 // [num_kv_heads * head_dim]
	BV []float32 // [num_kv_heads * head_dim]
	BO []float32 // [dim]

	// MLP projections (gated MLP / SwiGLU)
	WGate     []byte // gate_proj [interm, dim]
	WGateType uint32
	WUp       []byte // up_proj [interm, dim]
	WUpType   uint32
	WDown     []byte // down_proj [dim, interm]
	WDownType uint32
}

// LlamaState holds runtime buffers and KV cache
type LlamaState struct {
	X      []float32 // current hidden state [dim]
	XB     []float32 // buffer after norm [dim]
	XB2    []float32 // second buffer [dim]
	HB     []float32 // MLP hidden buffer [interm]
	HB2    []float32 // MLP gate buffer [interm]
	Q      []float32 // query [n_heads * head_dim]
	K      []float32 // key [n_kv_heads * head_dim]
	V      []float32 // value [n_kv_heads * head_dim]
	Att    []float32 // attention scores [n_heads * seq_len]
	Logits []float32 // output logits [vocab]

	// KV cache [layer * seq_len * kv_dim]
	KeyCache   []float32
	ValueCache []float32

	// RoPE precomputed
	CosCache []float32 // [seq_len * head_dim/2]
	SinCache []float32

	// Reusable embedding buffer (avoids allocation per Forward call)
	EmbBuf []float32

	// Position tracking
	Pos int
}

// LoadLlamaModel builds a LlamaModel from a parsed GGUF file
func LoadLlamaModel(gguf *GGUFFile) (*LlamaModel, error) {
	m := &GGUFMetadata{}
	*m = gguf.Meta

	cfg := LlamaConfig{
		NumLayers:     m.NumLayers,
		EmbedDim:      m.EmbedDim,
		NumHeads:      m.NumHeads,
		NumKVHeads:    m.NumKVHeads,
		HeadDim:       m.HeadDim,
		VocabSize:     m.VocabSize,
		SeqLen:        m.SeqLen,
		IntermSize:    m.IntermSize,
		RMSNormEps:    m.RMSNormEps,
		RopeTheta:     m.RopeTheta,
		QKNorm:        m.QKNorm,
		RopeConjugate: m.RopeConjugate,
	}

	if cfg.HeadDim == 0 && cfg.NumHeads > 0 {
		cfg.HeadDim = cfg.EmbedDim / cfg.NumHeads
	}

	// Cap sequence length to save memory
	if cfg.SeqLen > 2048 {
		fmt.Printf("[model] capping seq_len from %d to 2048\n", cfg.SeqLen)
		cfg.SeqLen = 2048
	}

	// Load weights
	w, err := loadWeights(gguf, &cfg)
	if err != nil {
		return nil, fmt.Errorf("load weights: %w", err)
	}

	// Allocate state
	state := allocState(&cfg)
	precomputeRoPE(&state, &cfg)

	model := &LlamaModel{
		Config:  cfg,
		Weights: *w,
		State:   state,
	}

	hasBias := w.Layers[0].BQ != nil
	fmt.Printf("[model] loaded: %d layers, %d dim, %d heads, %d kv_heads, %d vocab, bias=%v\n",
		cfg.NumLayers, cfg.EmbedDim, cfg.NumHeads, cfg.NumKVHeads, cfg.VocabSize, hasBias)
	if cfg.QKNorm || cfg.RopeConjugate {
		fmt.Printf("[model] nanollama flags: qk_norm=%v rope_conjugate=%v\n", cfg.QKNorm, cfg.RopeConjugate)
	}

	return model, nil
}

// loadWeights maps GGUF tensors to LlamaWeights
func loadWeights(gguf *GGUFFile, cfg *LlamaConfig) (*LlamaWeights, error) {
	w := &LlamaWeights{}

	// Token embedding
	emb, embInfo, err := gguf.GetTensor("token_embd.weight")
	if err != nil {
		return nil, fmt.Errorf("token_embd.weight: %w", err)
	}
	w.TokenEmbed = emb
	w.TokenEmbType = embInfo.Type

	// Output norm
	w.OutputNorm, err = getF32Tensor(gguf, "output_norm.weight", cfg.EmbedDim)
	if err != nil {
		return nil, fmt.Errorf("output_norm.weight: %w", err)
	}

	// Output (LM head) — might be tied to embedding
	outData, outInfo, err := gguf.GetTensor("output.weight")
	if err != nil {
		// Not found — use tied embeddings
		outData = w.TokenEmbed
		outInfo = embInfo
		fmt.Printf("[model] output.weight not found, using tied embeddings\n")
	} else {
		fmt.Printf("[model] output.weight: type=%d\n", outInfo.Type)
	}
	w.Output = outData
	w.OutputType = outInfo.Type

	// Per-layer weights
	w.Layers = make([]LlamaLayerWeights, cfg.NumLayers)
	for i := 0; i < cfg.NumLayers; i++ {
		prefix := fmt.Sprintf("blk.%d.", i)
		l := &w.Layers[i]

		// Attention norm
		l.AttnNorm, err = getF32Tensor(gguf, prefix+"attn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_norm: %w", i, err)
		}

		// FFN norm
		l.FFNNorm, err = getF32Tensor(gguf, prefix+"ffn_norm.weight", cfg.EmbedDim)
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_norm: %w", i, err)
		}

		// Attention projections
		l.WQ, l.WQType, err = getRawTensor(gguf, prefix+"attn_q.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_q: %w", i, err)
		}
		l.WK, l.WKType, err = getRawTensor(gguf, prefix+"attn_k.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_k: %w", i, err)
		}
		l.WV, l.WVType, err = getRawTensor(gguf, prefix+"attn_v.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_v: %w", i, err)
		}
		l.WO, l.WOType, err = getRawTensor(gguf, prefix+"attn_output.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d attn_output: %w", i, err)
		}

		// Attention biases (optional)
		l.BQ, _ = getF32TensorOptional(gguf, prefix+"attn_q.bias", cfg.NumHeads*cfg.HeadDim)
		l.BK, _ = getF32TensorOptional(gguf, prefix+"attn_k.bias", cfg.NumKVHeads*cfg.HeadDim)
		l.BV, _ = getF32TensorOptional(gguf, prefix+"attn_v.bias", cfg.NumKVHeads*cfg.HeadDim)
		l.BO, _ = getF32TensorOptional(gguf, prefix+"attn_output.bias", cfg.EmbedDim)

		// MLP projections (gated MLP / SwiGLU)
		l.WGate, l.WGateType, err = getRawTensor(gguf, prefix+"ffn_gate.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_gate: %w", i, err)
		}
		l.WUp, l.WUpType, err = getRawTensor(gguf, prefix+"ffn_up.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_up: %w", i, err)
		}
		l.WDown, l.WDownType, err = getRawTensor(gguf, prefix+"ffn_down.weight")
		if err != nil {
			return nil, fmt.Errorf("layer %d ffn_down: %w", i, err)
		}
	}

	return w, nil
}

// getF32Tensor loads a tensor and dequantizes to float32
func getF32Tensor(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, err
	}

	switch info.Type {
	case ggmlTypeF32:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			out[i] = math.Float32frombits(
				uint32(data[i*4]) | uint32(data[i*4+1])<<8 |
					uint32(data[i*4+2])<<16 | uint32(data[i*4+3])<<24)
		}
		return out, nil
	case ggmlTypeF16:
		out := make([]float32, expectedSize)
		for i := 0; i < expectedSize; i++ {
			h := uint16(data[i*2]) | uint16(data[i*2+1])<<8
			out[i] = half2float(h)
		}
		return out, nil
	case ggmlTypeQ4_0:
		return DequantQ4_0(data, expectedSize), nil
	case ggmlTypeQ5_0:
		return DequantQ5_0(data, expectedSize), nil
	case ggmlTypeQ8_0:
		return DequantQ8_0(data, expectedSize), nil
	case ggmlTypeQ4_K:
		return DequantQ4_K(data, expectedSize), nil
	case ggmlTypeQ6_K:
		return DequantQ6_K(data, expectedSize), nil
	default:
		return nil, fmt.Errorf("unsupported tensor type %d for %s", info.Type, name)
	}
}

// getF32TensorOptional loads a tensor if it exists, returns nil if not found
func getF32TensorOptional(gguf *GGUFFile, name string, expectedSize int) ([]float32, error) {
	_, _, err := gguf.GetTensor(name)
	if err != nil {
		return nil, nil // not found — not an error
	}
	return getF32Tensor(gguf, name, expectedSize)
}

// getRawTensor returns raw bytes + type for a tensor
func getRawTensor(gguf *GGUFFile, name string) ([]byte, uint32, error) {
	data, info, err := gguf.GetTensor(name)
	if err != nil {
		return nil, 0, err
	}
	return data, info.Type, nil
}

// allocState allocates all runtime buffers
func allocState(cfg *LlamaConfig) LlamaState {
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	return LlamaState{
		X:          make([]float32, cfg.EmbedDim),
		XB:         make([]float32, cfg.EmbedDim),
		XB2:        make([]float32, cfg.EmbedDim),
		HB:         make([]float32, cfg.IntermSize),
		HB2:        make([]float32, cfg.IntermSize),
		Q:          make([]float32, cfg.NumHeads*cfg.HeadDim),
		K:          make([]float32, kvDim),
		V:          make([]float32, kvDim),
		Att:        make([]float32, cfg.NumHeads*cfg.SeqLen),
		Logits:     make([]float32, cfg.VocabSize),
		KeyCache:   make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		ValueCache: make([]float32, cfg.NumLayers*cfg.SeqLen*kvDim),
		CosCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		SinCache:   make([]float32, cfg.SeqLen*(cfg.HeadDim/2)),
		EmbBuf:     make([]float32, cfg.EmbedDim),
	}
}

// precomputeRoPE fills cos/sin caches for rotary position encoding
func precomputeRoPE(s *LlamaState, cfg *LlamaConfig) {
	half := cfg.HeadDim / 2
	theta := float64(cfg.RopeTheta)

	for pos := 0; pos < cfg.SeqLen; pos++ {
		for i := 0; i < half; i++ {
			freq := 1.0 / math.Pow(theta, float64(2*i)/float64(cfg.HeadDim))
			angle := float64(pos) * freq
			s.CosCache[pos*half+i] = float32(math.Cos(angle))
			s.SinCache[pos*half+i] = float32(math.Sin(angle))
		}
	}
}

// matmulDispatch dispatches to the right matmul based on tensor type
func matmulDispatch(out []float32, w []byte, wtype uint32, x []float32, rows, cols int) {
	switch wtype {
	case ggmlTypeQ4_0:
		MatMulQ4_0(out, w, x, rows, cols)
	case ggmlTypeQ5_0:
		MatMulQ5_0(out, w, x, rows, cols)
	case ggmlTypeQ8_0:
		MatMulQ8_0(out, w, x, rows, cols)
	case ggmlTypeF16:
		MatMulF16(out, w, x, rows, cols)
	case ggmlTypeF32:
		f32 := make([]float32, len(w)/4)
		for i := range f32 {
			f32[i] = math.Float32frombits(
				uint32(w[i*4]) | uint32(w[i*4+1])<<8 |
					uint32(w[i*4+2])<<16 | uint32(w[i*4+3])<<24)
		}
		MatMulF32(out, f32, x, rows, cols)
	case ggmlTypeQ4_K:
		MatMulQ4_K(out, w, x, rows, cols)
	case ggmlTypeQ6_K:
		MatMulQ6_K(out, w, x, rows, cols)
	default:
		fmt.Printf("[model] WARNING: unsupported matmul type %d for %dx%d\n", wtype, rows, cols)
	}
}

// embedLookupInto extracts an embedding row into a pre-allocated buffer (zero alloc)
func embedLookupInto(out []float32, data []byte, dtype uint32, token, dim int) {
	switch dtype {
	case ggmlTypeQ4_0:
		blocksPerRow := dim / q4BlockSize
		bytesPerRow := blocksPerRow * q4BytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4BytesPerBlock
			DequantQ4_0Block(data[blockOff:blockOff+q4BytesPerBlock], out[b*q4BlockSize:])
		}
	case ggmlTypeQ5_0:
		blocksPerRow := dim / q50BlockSize
		bytesPerRow := blocksPerRow * q50BytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q50BytesPerBlock
			DequantQ5_0Block(data[blockOff:blockOff+q50BytesPerBlock], out[b*q50BlockSize:])
		}
	case ggmlTypeQ8_0:
		blocksPerRow := dim / q8BlockSize
		bytesPerRow := blocksPerRow * q8BytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q8BytesPerBlock
			DequantQ8_0Block(data[blockOff:blockOff+q8BytesPerBlock], out[b*q8BlockSize:])
		}
	case ggmlTypeQ4_K:
		blocksPerRow := dim / q4kBlockSize
		bytesPerRow := blocksPerRow * q4kBytesPerBlock
		rowOff := token * bytesPerRow
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4kBytesPerBlock
			DequantQ4_KBlock(data[blockOff:blockOff+q4kBytesPerBlock], out[b*q4kBlockSize:])
		}
	case ggmlTypeQ6_K:
		blocksPerRow := dim / q6kBlockSize
		bytesPerRow := blocksPerRow * q6kBytesPerBlock
		rowOff := token * bytesPerRow
		copy(out, DequantQ6_K(data[rowOff:rowOff+bytesPerRow], dim))
	case ggmlTypeF16:
		off := token * dim * 2
		for i := 0; i < dim; i++ {
			h := uint16(data[off+i*2]) | uint16(data[off+i*2+1])<<8
			out[i] = half2float(h)
		}
	case ggmlTypeF32:
		off := token * dim * 4
		for i := 0; i < dim; i++ {
			out[i] = math.Float32frombits(
				uint32(data[off+i*4]) | uint32(data[off+i*4+1])<<8 |
					uint32(data[off+i*4+2])<<16 | uint32(data[off+i*4+3])<<24)
		}
	default:
		for i := 0; i < dim; i++ {
			out[i] = 0
		}
	}
}

// applyRoPE applies standard rotary position encoding
func applyRoPE(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		x0 := vec[i]
		x1 := vec[i+half]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[i] = x0*c - x1*si
		vec[i+half] = x0*si + x1*c
	}
}

// applyRoPEConjugate applies conjugate rotary position encoding.
// nanollama convention: (x0*cos + x1*sin, -x0*sin + x1*cos)
func applyRoPEConjugate(vec []float32, pos int, s *LlamaState, headDim int) {
	half := headDim / 2
	cacheOff := pos * half

	for i := 0; i < half; i++ {
		x0 := vec[i]
		x1 := vec[i+half]
		c := s.CosCache[cacheOff+i]
		si := s.SinCache[cacheOff+i]
		vec[i] = x0*c + x1*si
		vec[i+half] = -x0*si + x1*c
	}
}

// addBias adds bias vector to output (no-op if bias is nil)
func addBias(out []float32, bias []float32) {
	if bias == nil {
		return
	}
	for i := range bias {
		out[i] += bias[i]
	}
}

// Forward runs one token through the transformer
func (m *LlamaModel) Forward(token int, pos int) {
	cfg := &m.Config
	w := &m.Weights
	s := &m.State
	dim := cfg.EmbedDim
	kvDim := cfg.NumKVHeads * cfg.HeadDim
	hd := cfg.HeadDim
	headGroupSize := cfg.NumHeads / cfg.NumKVHeads

	// 1. Token embedding lookup (zero-alloc: reuses s.EmbBuf)
	embedLookupInto(s.EmbBuf, w.TokenEmbed, w.TokenEmbType, token, dim)

	// 1.5. Gamma injection: embed[token] += γ[token]
	if m.Gamma != nil {
		m.Gamma.ApplyToEmbedding(s.EmbBuf, token)
	}

	copy(s.X, s.EmbBuf)

	// Pre-compute attention scale
	attnScale := float32(1.0 / math.Sqrt(float64(hd)))

	// 2. Transformer layers
	for layer := 0; layer < cfg.NumLayers; layer++ {
		l := &w.Layers[layer]

		// Attention pre-norm
		RMSNormInto(s.XB, s.X, l.AttnNorm, cfg.RMSNormEps)

		// Q, K, V projections
		matmulDispatch(s.Q, l.WQ, l.WQType, s.XB, cfg.NumHeads*hd, dim)
		matmulDispatch(s.K, l.WK, l.WKType, s.XB, cfg.NumKVHeads*hd, dim)
		matmulDispatch(s.V, l.WV, l.WVType, s.XB, cfg.NumKVHeads*hd, dim)

		// Add bias (no-op if nil)
		addBias(s.Q, l.BQ)
		addBias(s.K, l.BK)
		addBias(s.V, l.BV)

		// RoPE on Q and K (conjugate for nanollama, standard for Qwen/Llama)
		ropeFunc := applyRoPE
		if cfg.RopeConjugate {
			ropeFunc = applyRoPEConjugate
		}
		for h := 0; h < cfg.NumHeads; h++ {
			ropeFunc(s.Q[h*hd:(h+1)*hd], pos, s, hd)
		}
		for h := 0; h < cfg.NumKVHeads; h++ {
			ropeFunc(s.K[h*hd:(h+1)*hd], pos, s, hd)
		}

		// QK-norm: normalize Q and K per-head after RoPE (nanollama)
		if cfg.QKNorm {
			for h := 0; h < cfg.NumHeads; h++ {
				RMSNormBare(s.Q[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
			for h := 0; h < cfg.NumKVHeads; h++ {
				RMSNormBare(s.K[h*hd:(h+1)*hd], cfg.RMSNormEps)
			}
		}

		// Store K, V in cache
		cacheOff := layer*cfg.SeqLen*kvDim + pos*kvDim
		copy(s.KeyCache[cacheOff:cacheOff+kvDim], s.K[:kvDim])
		copy(s.ValueCache[cacheOff:cacheOff+kvDim], s.V[:kvDim])

		// Multi-head attention with GQA
		for h := 0; h < cfg.NumHeads; h++ {
			kvh := h / headGroupSize
			qh := s.Q[h*hd : (h+1)*hd]
			att := s.Att[h*cfg.SeqLen : h*cfg.SeqLen+pos+1]

			// QK dot products
			for t := 0; t <= pos; t++ {
				kOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				var dot float32
				for d := 0; d < hd; d++ {
					dot += qh[d] * s.KeyCache[kOff+d]
				}
				att[t] = dot * attnScale
			}

			// Softmax
			Softmax(att, pos+1)

			// Weighted sum of values → XB2
			xbSlice := s.XB2[h*hd : (h+1)*hd]
			for d := 0; d < hd; d++ {
				xbSlice[d] = 0
			}
			for t := 0; t <= pos; t++ {
				a := att[t]
				vOff := layer*cfg.SeqLen*kvDim + t*kvDim + kvh*hd
				for d := 0; d < hd; d++ {
					xbSlice[d] += a * s.ValueCache[vOff+d]
				}
			}
		}

		// Output projection: XB = WO × XB2 + bias, then residual
		matmulDispatch(s.XB, l.WO, l.WOType, s.XB2, dim, dim)
		addBias(s.XB, l.BO)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}

		// MLP: pre-norm
		RMSNormInto(s.XB, s.X, l.FFNNorm, cfg.RMSNormEps)

		// Gated MLP: gate_proj and up_proj
		matmulDispatch(s.HB, l.WGate, l.WGateType, s.XB, cfg.IntermSize, dim)
		matmulDispatch(s.HB2, l.WUp, l.WUpType, s.XB, cfg.IntermSize, dim)

		// SiLU(gate) * up
		for i := 0; i < cfg.IntermSize; i++ {
			s.HB[i] = SiLU(s.HB[i]) * s.HB2[i]
		}

		// down_proj + residual
		matmulDispatch(s.XB, l.WDown, l.WDownType, s.HB, dim, cfg.IntermSize)
		for i := 0; i < dim; i++ {
			s.X[i] += s.XB[i]
		}
	}

	// 3. Final norm
	RMSNorm(s.X, w.OutputNorm, cfg.RMSNormEps)

	// 4. LM head → logits
	matmulDispatch(s.Logits, w.Output, w.OutputType, s.X, cfg.VocabSize, dim)
}

// Reset clears KV cache and position for new generation
func (m *LlamaModel) Reset() {
	for i := range m.State.KeyCache {
		m.State.KeyCache[i] = 0
	}
	for i := range m.State.ValueCache {
		m.State.ValueCache[i] = 0
	}
	m.State.Pos = 0
}
