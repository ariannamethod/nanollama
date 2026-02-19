package main

// tokenizer.go — BPE tokenizer from GGUF metadata
//
// Supports two modes:
//   1. SentencePiece BPE (LLaMA/TinyLlama) — ▁ prefix, score-based merges
//   2. GPT-2 byte-level BPE (Qwen2.5) — byte-to-unicode mapping, merge-rank BPE
//
// Mode is auto-detected from tokenizer.ggml.model in GGUF metadata.

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Tokenizer handles BPE encoding/decoding (SentencePiece or GPT-2)
type Tokenizer struct {
	Vocab          []string
	Scores         []float32
	Types          []int32
	VocabSize      int
	BosID          int
	EosID          int
	AddSpacePrefix bool
	IsGPT2         bool // true for GPT-2 byte-level BPE

	// Lookup table for encoding
	tokenToID map[string]int
	// Byte fallback tokens (SentencePiece style <0xNN>)
	byteTokens [256]int

	// GPT-2 byte-level encoding tables
	byteToUnicode [256]rune
	unicodeToByte map[rune]byte

	// GPT-2 BPE merge ranks
	mergeRank map[string]int

	// Special tokens that should be matched as whole units
	specialTokens map[string]int

	// GPT-2/Qwen2 pre-tokenizer regex
	preTokenRe *regexp.Regexp
}

// buildGPT2ByteTable builds the GPT-2 bytes_to_unicode mapping
func buildGPT2ByteTable() (byteToUni [256]rune, uniToByte map[rune]byte) {
	uniToByte = make(map[rune]byte, 256)
	n := 0
	for b := 0; b < 256; b++ {
		var r rune
		if (b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255) {
			r = rune(b)
		} else {
			r = rune(256 + n)
			n++
		}
		byteToUni[b] = r
		uniToByte[r] = byte(b)
	}
	return
}

// NewTokenizer creates a tokenizer from GGUF metadata
func NewTokenizer(meta *GGUFMetadata) *Tokenizer {
	isGPT2 := meta.TokenizerModel == "gpt2"

	t := &Tokenizer{
		Vocab:     meta.TokenList,
		Scores:    meta.TokenScores,
		Types:     meta.TokenTypes,
		VocabSize: meta.VocabSize,
		BosID:     meta.BosID,
		EosID:     meta.EosID,
		IsGPT2:   isGPT2,
	}

	if isGPT2 {
		t.AddSpacePrefix = false
		t.byteToUnicode, t.unicodeToByte = buildGPT2ByteTable()
		t.preTokenRe = regexp.MustCompile(
			`(?i:'s|'t|'re|'ve|'m|'ll|'d)` +
				`|[^\r\n\p{L}\p{N}]?\p{L}+` +
				`|\p{N}{1,3}` +
				`| ?[^\s\p{L}\p{N}]+[\r\n]*` +
				`|\s*[\r\n]+` +
				`|\s+`)
	} else {
		t.AddSpacePrefix = meta.AddSpacePrefix
	}

	// Build lookup table
	t.tokenToID = make(map[string]int, t.VocabSize)
	for i, tok := range t.Vocab {
		t.tokenToID[tok] = i
	}

	// Map byte fallback tokens (SentencePiece style)
	for i := 0; i < 256; i++ {
		name := fmt.Sprintf("<0x%02X>", i)
		if id, ok := t.tokenToID[name]; ok {
			t.byteTokens[i] = id
		} else {
			t.byteTokens[i] = -1
		}
	}

	// Build special tokens map (control tokens)
	t.specialTokens = make(map[string]int)
	if t.Types != nil {
		for i, typ := range t.Types {
			if typ == 3 && i < len(t.Vocab) {
				token := t.Vocab[i]
				if len(token) > 2 {
					t.specialTokens[token] = i
				}
			}
		}
		fmt.Printf("[tokenizer] %d special tokens registered\n", len(t.specialTokens))
	}

	// GPT-2 BPE: build merge rank map
	if isGPT2 && len(meta.TokenMerges) > 0 {
		t.mergeRank = make(map[string]int, len(meta.TokenMerges))
		for i, merge := range meta.TokenMerges {
			t.mergeRank[merge] = i
		}
		fmt.Printf("[tokenizer] GPT-2 BPE mode, %d merges\n", len(meta.TokenMerges))
	}

	fmt.Printf("[tokenizer] vocab=%d bos=%d eos=%d gpt2=%v add_space_prefix=%v\n",
		t.VocabSize, t.BosID, t.EosID, t.IsGPT2, t.AddSpacePrefix)
	return t
}

// Encode converts text to token IDs using BPE
func (t *Tokenizer) Encode(text string, addBos bool) []int {
	var tokens []int

	if addBos && t.BosID >= 0 {
		tokens = append(tokens, t.BosID)
	}

	if len(text) == 0 {
		return tokens
	}

	segments := t.splitOnSpecialTokens(text)
	for _, seg := range segments {
		if id, ok := t.specialTokens[seg]; ok {
			tokens = append(tokens, id)
		} else if t.IsGPT2 {
			tokens = append(tokens, t.encodeGPT2(seg)...)
		} else {
			tokens = append(tokens, t.encodeSentencePiece(seg)...)
		}
	}

	return tokens
}

// splitOnSpecialTokens splits text into segments, preserving special tokens
func (t *Tokenizer) splitOnSpecialTokens(text string) []string {
	if len(t.specialTokens) == 0 {
		return []string{text}
	}

	var segments []string
	remaining := text

	for len(remaining) > 0 {
		bestPos := -1
		bestLen := 0
		bestToken := ""

		for token := range t.specialTokens {
			pos := strings.Index(remaining, token)
			if pos >= 0 && (bestPos < 0 || pos < bestPos || (pos == bestPos && len(token) > bestLen)) {
				bestPos = pos
				bestLen = len(token)
				bestToken = token
			}
		}

		if bestPos < 0 {
			if len(remaining) > 0 {
				segments = append(segments, remaining)
			}
			break
		}

		if bestPos > 0 {
			segments = append(segments, remaining[:bestPos])
		}
		segments = append(segments, bestToken)
		remaining = remaining[bestPos+bestLen:]
	}

	return segments
}

// encodeSentencePiece does SentencePiece BPE encoding
func (t *Tokenizer) encodeSentencePiece(text string) []int {
	if t.AddSpacePrefix && len(text) > 0 && text[0] != ' ' {
		text = " " + text
	}
	text = strings.ReplaceAll(text, " ", "\u2581")

	symbols := t.initialTokenizeSP(text)
	symbols = t.bpeMerge(symbols)
	return t.symbolsToIDs(symbols)
}

// encodeGPT2 does GPT-2 byte-level BPE encoding
func (t *Tokenizer) encodeGPT2(text string) []int {
	chunks := t.preTokenRe.FindAllString(text, -1)
	if len(chunks) == 0 {
		return nil
	}

	var allTokens []int
	for _, chunk := range chunks {
		rawBytes := []byte(chunk)
		symbols := make([]string, len(rawBytes))
		for i, b := range rawBytes {
			symbols[i] = string(t.byteToUnicode[b])
		}
		symbols = t.bpeMergeGPT2(symbols)
		allTokens = append(allTokens, t.symbolsToIDs(symbols)...)
	}
	return allTokens
}

// bpeMergeGPT2 applies BPE merging using merge rank ordering
func (t *Tokenizer) bpeMergeGPT2(symbols []string) []string {
	for {
		bestRank := len(t.mergeRank) + 1
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			pair := symbols[i] + " " + symbols[i+1]
			if rank, ok := t.mergeRank[pair]; ok {
				if rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

// bpeMerge applies greedy BPE merging using token scores
func (t *Tokenizer) bpeMerge(symbols []string) []string {
	for {
		bestScore := float32(-1e30)
		bestIdx := -1

		for i := 0; i < len(symbols)-1; i++ {
			merged := symbols[i] + symbols[i+1]
			if id, ok := t.tokenToID[merged]; ok {
				score := t.Scores[id]
				if score > bestScore {
					bestScore = score
					bestIdx = i
				}
			}
		}

		if bestIdx < 0 {
			break
		}

		merged := symbols[bestIdx] + symbols[bestIdx+1]
		newSymbols := make([]string, 0, len(symbols)-1)
		newSymbols = append(newSymbols, symbols[:bestIdx]...)
		newSymbols = append(newSymbols, merged)
		newSymbols = append(newSymbols, symbols[bestIdx+2:]...)
		symbols = newSymbols
	}
	return symbols
}

// symbolsToIDs converts BPE symbols to token IDs with byte fallback
func (t *Tokenizer) symbolsToIDs(symbols []string) []int {
	var tokens []int
	for _, sym := range symbols {
		if id, ok := t.tokenToID[sym]; ok {
			tokens = append(tokens, id)
		} else {
			for _, b := range []byte(sym) {
				if t.byteTokens[b] >= 0 {
					tokens = append(tokens, t.byteTokens[b])
				}
			}
		}
	}
	return tokens
}

// initialTokenizeSP splits text into initial symbols for SentencePiece BPE
func (t *Tokenizer) initialTokenizeSP(text string) []string {
	var symbols []string

	runes := []rune(text)
	i := 0
	for i < len(runes) {
		ch := string(runes[i])
		if _, ok := t.tokenToID[ch]; ok {
			symbols = append(symbols, ch)
			i++
			continue
		}

		for _, b := range []byte(string(runes[i])) {
			byteStr := fmt.Sprintf("<0x%02X>", b)
			symbols = append(symbols, byteStr)
		}
		i++
	}

	return symbols
}

// Decode converts token IDs back to text
func (t *Tokenizer) Decode(ids []int) string {
	var sb strings.Builder
	for _, id := range ids {
		if id < 0 || id >= t.VocabSize {
			continue
		}
		piece := t.Vocab[id]

		if t.Types != nil && id < len(t.Types) && t.Types[id] == 3 {
			continue
		}

		if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
			var b byte
			fmt.Sscanf(piece, "<0x%02X>", &b)
			sb.WriteByte(b)
			continue
		}

		if t.IsGPT2 {
			for _, r := range piece {
				if b, ok := t.unicodeToByte[r]; ok {
					sb.WriteByte(b)
				} else {
					sb.WriteRune(r)
				}
			}
		} else {
			piece = strings.ReplaceAll(piece, "\u2581", " ")
			sb.WriteString(piece)
		}
	}

	result := sb.String()
	if !t.IsGPT2 && t.AddSpacePrefix && len(result) > 0 && result[0] == ' ' {
		result = result[1:]
	}
	return result
}

// DecodeToken decodes a single token ID
func (t *Tokenizer) DecodeToken(id int) string {
	if id < 0 || id >= t.VocabSize {
		return ""
	}
	piece := t.Vocab[id]

	if len(piece) == 6 && piece[0] == '<' && piece[1] == '0' && piece[2] == 'x' && piece[5] == '>' {
		var b byte
		fmt.Sscanf(piece, "<0x%02X>", &b)
		return string([]byte{b})
	}

	if t.IsGPT2 {
		var sb strings.Builder
		for _, r := range piece {
			if b, ok := t.unicodeToByte[r]; ok {
				sb.WriteByte(b)
			} else {
				sb.WriteRune(r)
			}
		}
		return sb.String()
	}

	piece = strings.ReplaceAll(piece, "\u2581", " ")
	return piece
}

// FindSpecialToken searches for a special token by name
func (t *Tokenizer) FindSpecialToken(name string) int {
	variants := []string{
		name,
		"<|" + name + "|>",
		"<" + name + ">",
	}
	for _, v := range variants {
		if id, ok := t.tokenToID[v]; ok {
			return id
		}
	}
	return -1
}

// DebugTokenize shows tokens for debugging
func (t *Tokenizer) DebugTokenize(text string) {
	ids := t.Encode(text, false)
	fmt.Printf("[tokenizer] '%s' -> %d tokens: ", text, len(ids))
	for _, id := range ids {
		if id >= 0 && id < t.VocabSize {
			fmt.Printf("[%d:'%s'] ", id, t.Vocab[id])
		}
	}
	fmt.Println()
}

// SortVocabByScore returns vocab indices sorted by score
func (t *Tokenizer) SortVocabByScore() []int {
	idx := make([]int, t.VocabSize)
	for i := range idx {
		idx[i] = i
	}
	sort.Slice(idx, func(i, j int) bool {
		return t.Scores[idx[i]] > t.Scores[idx[j]]
	})
	return idx
}
