package main

// gamma.go — Gamma Essence: personality extraction from embeddings
//
// γ lives in the embedding layer — it carries identity.
// θ = ε + γ (base model + personality = soul)
//
// Stored as sparse NPZ: only tokens where training changed embeddings.
//   indices.npy — int32, which token IDs have nonzero γ
//   values.npy  — float16/float32, [len(indices), embed_dim] — diff vectors
//
// Application: embed[token] += γ[token] (one-time at embedding lookup)
// Cost: 1 × embed_dim additions per token. Negligible.

import (
	"archive/zip"
	"fmt"
	"strings"
)

// GammaEssence holds the personality diff for embeddings
type GammaEssence struct {
	VocabSize int
	EmbedDim  int
	NumTokens int // how many tokens have nonzero gamma

	Indices   []int32   // token IDs with gamma
	Values    []float32 // [NumTokens × EmbedDim] — f32
	ValuesF16 []uint16  // [NumTokens × EmbedDim] — raw f16 (saves RAM)
	IsF16     bool

	// Fast lookup: token_id → index in Values
	IndexMap map[int32]int
}

// LoadGamma loads a gamma essence file from sparse NPZ format
func LoadGamma(path string) (*GammaEssence, error) {
	r, err := zip.OpenReader(path)
	if err != nil {
		return nil, fmt.Errorf("open gamma npz: %w", err)
	}
	defer r.Close()

	var indices []int32
	var valuesF16 []uint16
	var valuesF32 []float32
	var valShape [2]int
	var isF16 bool

	// Check if values.npy is f16
	for _, f := range r.File {
		if f.Name == "values.npy" {
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("open values.npy: %w", err)
			}
			hdr, err := readNpyHeader(rc)
			rc.Close()
			if err != nil {
				return nil, fmt.Errorf("peek values.npy: %w", err)
			}
			isF16 = strings.Contains(hdr, "'<f2'") || strings.Contains(hdr, "float16")
			break
		}
	}

	for _, f := range r.File {
		switch f.Name {
		case "indices.npy":
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("open indices.npy: %w", err)
			}
			indices, err = readNpyInt32(rc)
			rc.Close()
			if err != nil {
				return nil, fmt.Errorf("read indices.npy: %w", err)
			}

		case "values.npy":
			rc, err := f.Open()
			if err != nil {
				return nil, fmt.Errorf("open values.npy: %w", err)
			}
			if isF16 {
				var err2 error
				valuesF16, valShape, err2 = readNpyF16Raw(rc)
				rc.Close()
				if err2 != nil {
					return nil, fmt.Errorf("read values.npy f16: %w", err2)
				}
			} else {
				var err2 error
				valuesF32, valShape, err2 = readNpyFloat(rc)
				rc.Close()
				if err2 != nil {
					return nil, fmt.Errorf("read values.npy f32: %w", err2)
				}
			}
		}
	}

	if indices == nil || (valuesF16 == nil && valuesF32 == nil) {
		return nil, fmt.Errorf("gamma npz missing indices.npy or values.npy")
	}

	numTokens := valShape[0]
	embedDim := valShape[1]

	if len(indices) != numTokens {
		return nil, fmt.Errorf("indices len %d != values rows %d", len(indices), numTokens)
	}

	indexMap := make(map[int32]int, numTokens)
	var vocabSize int
	for i, idx := range indices {
		indexMap[idx] = i
		if int(idx)+1 > vocabSize {
			vocabSize = int(idx) + 1
		}
	}

	var ramMB float64
	if isF16 {
		ramMB = float64(len(valuesF16)*2) / 1024 / 1024
	} else {
		ramMB = float64(len(valuesF32)*4) / 1024 / 1024
	}
	dtype := "f32"
	if isF16 {
		dtype = "f16"
	}
	fmt.Printf("[gamma] loaded %s: %d/%d tokens, embed_dim=%d (%.1f MB RAM)\n",
		dtype, numTokens, vocabSize, embedDim, ramMB)

	return &GammaEssence{
		VocabSize: vocabSize,
		EmbedDim:  embedDim,
		NumTokens: numTokens,
		Indices:   indices,
		Values:    valuesF32,
		ValuesF16: valuesF16,
		IsF16:     isF16,
		IndexMap:  indexMap,
	}, nil
}

// ApplyToEmbedding adds gamma diff to the embedding of a token.
// embed must be [EmbedDim]. Modifies in place.
func (g *GammaEssence) ApplyToEmbedding(embed []float32, token int) {
	if g == nil {
		return
	}
	pos, ok := g.IndexMap[int32(token)]
	if !ok {
		return // this token has no gamma
	}
	off := pos * g.EmbedDim
	if g.IsF16 {
		for i := 0; i < g.EmbedDim; i++ {
			embed[i] += half2float(g.ValuesF16[off+i])
		}
	} else {
		for i := 0; i < g.EmbedDim; i++ {
			embed[i] += g.Values[off+i]
		}
	}
}
