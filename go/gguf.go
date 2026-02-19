package main

// gguf.go — GGUF file parser
//
// GGUF is llama.cpp's binary format. Structure:
//   Header: magic + version + tensor_count + metadata_count
//   Metadata: key-value pairs (vocab, config, etc.)
//   Tensor info: name + dims + type + offset
//   Alignment padding
//   Tensor data blob
//
// Supports F32, F16, Q4_0, Q4_1, Q5_0, Q8_0, Q4_K, Q6_K quantization.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"strings"
)

// GGUF constants
const (
	ggufMagic   = 0x46554747 // "GGUF" in LE
	ggufVersion = 3

	// GGUF value types
	ggufTypeUint8   = 0
	ggufTypeInt8    = 1
	ggufTypeUint16  = 2
	ggufTypeInt16   = 3
	ggufTypeUint32  = 4
	ggufTypeInt32   = 5
	ggufTypeFloat32 = 6
	ggufTypeBool    = 7
	ggufTypeString  = 8
	ggufTypeArray   = 9
	ggufTypeUint64  = 10
	ggufTypeInt64   = 11
	ggufTypeFloat64 = 12

	// GGML tensor types
	ggmlTypeF32  = 0
	ggmlTypeF16  = 1
	ggmlTypeQ4_0 = 2
	ggmlTypeQ4_1 = 3
	ggmlTypeQ5_0 = 6
	ggmlTypeQ5_1 = 7
	ggmlTypeQ8_0 = 8
	ggmlTypeQ8_1 = 9
	ggmlTypeQ2_K = 10
	ggmlTypeQ3_K = 11
	ggmlTypeQ4_K = 12
	ggmlTypeQ5_K = 13
	ggmlTypeQ6_K = 14
)

// GGUFMetadata holds parsed metadata
type GGUFMetadata struct {
	// Model architecture
	NumLayers    int
	EmbedDim     int
	NumHeads     int
	NumKVHeads   int
	HeadDim      int
	VocabSize    int
	SeqLen       int
	IntermSize   int // MLP intermediate size
	RMSNormEps   float32
	RopeTheta    float32
	RopeFreqBase float32

	// nanollama-specific flags
	QKNorm        bool // normalize Q,K with RMSNorm after RoPE (parameterless)
	RopeConjugate bool // conjugate RoPE convention: (x0*cos+x1*sin, -x0*sin+x1*cos)

	// Tokenizer
	TokenList      []string
	TokenScores    []float32
	TokenTypes     []int32
	TokenMerges    []string // GPT-2 BPE merge rules (empty for SentencePiece)
	TokenizerModel string   // "llama" (SentencePiece) or "gpt2" (byte-level BPE)
	BosID          int
	EosID          int
	AddSpacePrefix bool

	// Raw KV store
	KV map[string]interface{}
}

// GGUFTensorInfo describes a tensor in the file
type GGUFTensorInfo struct {
	Name   string
	NDims  uint32
	Dims   [4]uint64
	Type   uint32
	Offset uint64
}

// GGUFFile is a parsed GGUF file
type GGUFFile struct {
	Meta       GGUFMetadata
	Tensors    map[string]*GGUFTensorInfo
	TensorData []byte // mmap'd or read tensor data blob
	DataOffset int64  // offset where tensor data starts in file
}

func readString(r io.Reader) (string, error) {
	var length uint64
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		return "", err
	}
	if length > 1<<24 { // 16MB sanity limit
		return "", fmt.Errorf("string too long: %d", length)
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return "", err
	}
	return string(buf), nil
}

func readValue(r io.Reader, vtype uint32) (interface{}, error) {
	switch vtype {
	case ggufTypeUint8:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt8:
		var v int8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint16:
		var v uint16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt16:
		var v int16
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeUint32:
		var v uint32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt32:
		var v int32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat32:
		var v float32
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeBool:
		var v uint8
		err := binary.Read(r, binary.LittleEndian, &v)
		return v != 0, err
	case ggufTypeString:
		return readString(r)
	case ggufTypeUint64:
		var v uint64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeInt64:
		var v int64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeFloat64:
		var v float64
		err := binary.Read(r, binary.LittleEndian, &v)
		return v, err
	case ggufTypeArray:
		var elemType uint32
		if err := binary.Read(r, binary.LittleEndian, &elemType); err != nil {
			return nil, err
		}
		var count uint64
		if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
			return nil, err
		}
		if count > 1<<24 {
			return nil, fmt.Errorf("array too large: %d", count)
		}
		arr := make([]interface{}, count)
		for i := uint64(0); i < count; i++ {
			v, err := readValue(r, elemType)
			if err != nil {
				return nil, err
			}
			arr[i] = v
		}
		return arr, nil
	default:
		return nil, fmt.Errorf("unknown GGUF type: %d", vtype)
	}
}

// toInt converts GGUF metadata value to int
func toInt(v interface{}) int {
	switch x := v.(type) {
	case uint32:
		return int(x)
	case int32:
		return int(x)
	case uint64:
		return int(x)
	case int64:
		return int(x)
	case uint8:
		return int(x)
	case int8:
		return int(x)
	case uint16:
		return int(x)
	case int16:
		return int(x)
	default:
		return 0
	}
}

// toFloat32 converts GGUF metadata value to float32
func toFloat32(v interface{}) float32 {
	switch x := v.(type) {
	case float32:
		return x
	case float64:
		return float32(x)
	case uint32:
		return float32(x)
	case int32:
		return float32(x)
	default:
		return 0
	}
}

// ggmlBlockSize returns bytes per block for a tensor type
func ggmlBlockSize(t uint32) int {
	switch t {
	case ggmlTypeF32:
		return 4
	case ggmlTypeF16:
		return 2
	case ggmlTypeQ4_0:
		return 18 // 2 (fp16 scale) + 16 (32 x 4-bit values)
	case ggmlTypeQ4_1:
		return 20 // 2 (min) + 2 (scale) + 16 data
	case ggmlTypeQ8_0:
		return 34 // 2 (fp16 scale) + 32 (32 x 8-bit)
	case ggmlTypeQ5_0:
		return 22 // 2 (scale) + 4 (qh) + 16 (qs) per 32 elements
	case ggmlTypeQ6_K:
		return 210 // 128 (ql) + 64 (qh) + 16 (scales) + 2 (d) per 256 elements
	case ggmlTypeQ4_K:
		return 144 // 2 (d) + 2 (dmin) + 12 (scales) + 128 (qs) per 256 elements
	default:
		return 0
	}
}

// ggmlBlockElements returns the number of elements per block
func ggmlBlockElements(t uint32) int {
	switch t {
	case ggmlTypeF32, ggmlTypeF16:
		return 1
	case ggmlTypeQ4_K, ggmlTypeQ6_K:
		return 256 // k-quant super block
	default:
		return 32 // Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
	}
}

// tensorBytes returns total bytes for a tensor
func tensorBytes(info *GGUFTensorInfo) uint64 {
	nel := uint64(1)
	for i := uint32(0); i < info.NDims; i++ {
		nel *= info.Dims[i]
	}
	bs := uint64(ggmlBlockSize(info.Type))
	be := uint64(ggmlBlockElements(info.Type))
	if be == 0 {
		return 0
	}
	return (nel / be) * bs
}

// LoadGGUF loads a GGUF file
func LoadGGUF(path string) (*GGUFFile, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open GGUF: %w", err)
	}
	defer f.Close()

	// Read header
	var magic uint32
	if err := binary.Read(f, binary.LittleEndian, &magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if magic != ggufMagic {
		return nil, fmt.Errorf("bad magic: 0x%08X (expected 0x%08X)", magic, ggufMagic)
	}

	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version < 2 || version > 3 {
		return nil, fmt.Errorf("unsupported GGUF version: %d", version)
	}

	var tensorCount, metadataCount uint64
	if err := binary.Read(f, binary.LittleEndian, &tensorCount); err != nil {
		return nil, err
	}
	if err := binary.Read(f, binary.LittleEndian, &metadataCount); err != nil {
		return nil, err
	}

	fmt.Printf("[gguf] version=%d tensors=%d metadata=%d\n", version, tensorCount, metadataCount)

	// Read metadata
	kv := make(map[string]interface{})
	for i := uint64(0); i < metadataCount; i++ {
		key, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read metadata key %d: %w", i, err)
		}
		var vtype uint32
		if err := binary.Read(f, binary.LittleEndian, &vtype); err != nil {
			return nil, fmt.Errorf("read metadata type %d: %w", i, err)
		}
		val, err := readValue(f, vtype)
		if err != nil {
			return nil, fmt.Errorf("read metadata value '%s': %w", key, err)
		}
		kv[key] = val
	}

	// Read tensor infos
	tensors := make(map[string]*GGUFTensorInfo, tensorCount)
	for i := uint64(0); i < tensorCount; i++ {
		name, err := readString(f)
		if err != nil {
			return nil, fmt.Errorf("read tensor name %d: %w", i, err)
		}
		var ndims uint32
		if err := binary.Read(f, binary.LittleEndian, &ndims); err != nil {
			return nil, err
		}
		var dims [4]uint64
		for d := uint32(0); d < ndims; d++ {
			if err := binary.Read(f, binary.LittleEndian, &dims[d]); err != nil {
				return nil, err
			}
		}
		var ttype uint32
		if err := binary.Read(f, binary.LittleEndian, &ttype); err != nil {
			return nil, err
		}
		var offset uint64
		if err := binary.Read(f, binary.LittleEndian, &offset); err != nil {
			return nil, err
		}
		tensors[name] = &GGUFTensorInfo{
			Name:   name,
			NDims:  ndims,
			Dims:   dims,
			Type:   ttype,
			Offset: offset,
		}
	}

	// Current position = end of header/metadata/tensor_info
	headerEnd, err := f.Seek(0, io.SeekCurrent)
	if err != nil {
		return nil, err
	}

	// GGUF alignment = 32 bytes
	alignment := int64(32)
	dataOffset := ((headerEnd + alignment - 1) / alignment) * alignment

	// Read all tensor data
	fileInfo, err := f.Stat()
	if err != nil {
		return nil, err
	}
	dataSize := fileInfo.Size() - dataOffset
	if dataSize <= 0 {
		return nil, fmt.Errorf("no tensor data (dataOffset=%d, fileSize=%d)", dataOffset, fileInfo.Size())
	}

	fmt.Printf("[gguf] data offset=%d size=%.1f MB\n", dataOffset, float64(dataSize)/1024/1024)

	if _, err := f.Seek(dataOffset, io.SeekStart); err != nil {
		return nil, err
	}
	tensorData := make([]byte, dataSize)
	if _, err := io.ReadFull(f, tensorData); err != nil {
		return nil, fmt.Errorf("read tensor data: %w", err)
	}

	// Parse metadata into structured form
	meta := parseMetadata(kv)

	return &GGUFFile{
		Meta:       meta,
		Tensors:    tensors,
		TensorData: tensorData,
		DataOffset: dataOffset,
	}, nil
}

// parseMetadata extracts model config from GGUF KV pairs
func parseMetadata(kv map[string]interface{}) GGUFMetadata {
	meta := GGUFMetadata{
		KV:         kv,
		RMSNormEps: 1e-5,
		RopeTheta:  10000.0,
		BosID:      1,
		EosID:      2,
	}

	// Architecture prefix (usually "llama")
	arch := "llama"
	if v, ok := kv["general.architecture"]; ok {
		if s, ok := v.(string); ok {
			arch = s
		}
	}

	// Model dimensions
	if v, ok := kv[arch+".block_count"]; ok {
		meta.NumLayers = toInt(v)
	}
	if v, ok := kv[arch+".embedding_length"]; ok {
		meta.EmbedDim = toInt(v)
	}
	if v, ok := kv[arch+".attention.head_count"]; ok {
		meta.NumHeads = toInt(v)
	}
	if v, ok := kv[arch+".attention.head_count_kv"]; ok {
		meta.NumKVHeads = toInt(v)
	}
	if v, ok := kv[arch+".feed_forward_length"]; ok {
		meta.IntermSize = toInt(v)
	}
	if v, ok := kv[arch+".context_length"]; ok {
		meta.SeqLen = toInt(v)
	}
	if v, ok := kv[arch+".attention.layer_norm_rms_epsilon"]; ok {
		meta.RMSNormEps = toFloat32(v)
	}
	if v, ok := kv[arch+".rope.freq_base"]; ok {
		meta.RopeTheta = toFloat32(v)
	}

	// Derived
	if meta.NumHeads > 0 && meta.EmbedDim > 0 {
		meta.HeadDim = meta.EmbedDim / meta.NumHeads
	}
	if meta.NumKVHeads == 0 {
		meta.NumKVHeads = meta.NumHeads // MHA fallback
	}

	// nanollama-specific flags
	if v, ok := kv["nanollama.qk_norm"]; ok {
		if b, ok := v.(bool); ok {
			meta.QKNorm = b
		}
	}
	if v, ok := kv["nanollama.rope_conjugate"]; ok {
		if b, ok := v.(bool); ok {
			meta.RopeConjugate = b
		}
	}

	// Tokenizer model type
	meta.TokenizerModel = "llama" // default: SentencePiece
	if v, ok := kv["tokenizer.ggml.model"]; ok {
		if s, ok := v.(string); ok {
			meta.TokenizerModel = s
		}
	}

	// Tokenizer
	if v, ok := kv["tokenizer.ggml.tokens"]; ok {
		if arr, ok := v.([]interface{}); ok {
			meta.TokenList = make([]string, len(arr))
			for i, tok := range arr {
				if s, ok := tok.(string); ok {
					meta.TokenList[i] = s
				}
			}
			meta.VocabSize = len(meta.TokenList)
		}
	}
	if v, ok := kv["tokenizer.ggml.scores"]; ok {
		if arr, ok := v.([]interface{}); ok {
			meta.TokenScores = make([]float32, len(arr))
			for i, s := range arr {
				meta.TokenScores[i] = toFloat32(s)
			}
		}
	}
	if v, ok := kv["tokenizer.ggml.token_type"]; ok {
		if arr, ok := v.([]interface{}); ok {
			meta.TokenTypes = make([]int32, len(arr))
			for i, t := range arr {
				meta.TokenTypes[i] = int32(toInt(t))
			}
		}
	}
	if v, ok := kv["tokenizer.ggml.bos_token_id"]; ok {
		meta.BosID = toInt(v)
	}
	if v, ok := kv["tokenizer.ggml.eos_token_id"]; ok {
		meta.EosID = toInt(v)
	}
	// BPE merges (GPT-2 style tokenizers)
	if v, ok := kv["tokenizer.ggml.merges"]; ok {
		if arr, ok := v.([]interface{}); ok {
			meta.TokenMerges = make([]string, len(arr))
			for i, m := range arr {
				if s, ok := m.(string); ok {
					meta.TokenMerges[i] = s
				}
			}
		}
	}

	// Default: add space prefix (standard SentencePiece behavior)
	meta.AddSpacePrefix = true
	if v, ok := kv["tokenizer.ggml.add_space_prefix"]; ok {
		switch val := v.(type) {
		case bool:
			meta.AddSpacePrefix = val
		case uint8:
			meta.AddSpacePrefix = val != 0
		case int:
			meta.AddSpacePrefix = val != 0
		case uint32:
			meta.AddSpacePrefix = val != 0
		}
	}

	fmt.Printf("[gguf] arch=%s layers=%d dim=%d heads=%d kv_heads=%d head_dim=%d\n",
		arch, meta.NumLayers, meta.EmbedDim, meta.NumHeads, meta.NumKVHeads, meta.HeadDim)
	fmt.Printf("[gguf] vocab=%d seq_len=%d ffn=%d rope_theta=%.1f tokenizer=%s\n",
		meta.VocabSize, meta.SeqLen, meta.IntermSize, meta.RopeTheta, meta.TokenizerModel)
	if len(meta.TokenMerges) > 0 {
		fmt.Printf("[gguf] BPE merges=%d\n", len(meta.TokenMerges))
	}

	return meta
}

// GetTensor returns raw bytes for a named tensor
func (g *GGUFFile) GetTensor(name string) ([]byte, *GGUFTensorInfo, error) {
	info, ok := g.Tensors[name]
	if !ok {
		return nil, nil, fmt.Errorf("tensor not found: %s", name)
	}
	size := tensorBytes(info)
	start := info.Offset
	end := start + size
	if end > uint64(len(g.TensorData)) {
		return nil, nil, fmt.Errorf("tensor %s out of bounds: %d + %d > %d",
			name, start, size, len(g.TensorData))
	}
	return g.TensorData[start:end], info, nil
}

// FindTensor searches for a tensor by substring match
func (g *GGUFFile) FindTensor(substr string) (*GGUFTensorInfo, bool) {
	for name, info := range g.Tensors {
		if strings.Contains(name, substr) {
			return info, true
		}
	}
	return nil, false
}

// ListTensors prints all tensors (debug)
func (g *GGUFFile) ListTensors() {
	for name, info := range g.Tensors {
		size := tensorBytes(info)
		fmt.Printf("  %-50s  type=%d  dims=[", name, info.Type)
		for d := uint32(0); d < info.NDims; d++ {
			if d > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%d", info.Dims[d])
		}
		fmt.Printf("]  %.2f MB\n", float64(size)/1024/1024)
	}
}

// half2floatLUT is a precomputed lookup table for all 65536 fp16 values.
// 256KB, fits in L2 cache. Eliminates branching in the hottest matmul paths.
var half2floatLUT [65536]float32

func init() {
	for h := 0; h < 65536; h++ {
		sign := uint32(h>>15) & 1
		exp := uint32(h>>10) & 0x1F
		mant := uint32(h & 0x3FF)

		var f uint32
		if exp == 0 {
			if mant == 0 {
				f = sign << 31
			} else {
				e := uint32(1)
				for mant&0x400 == 0 {
					mant <<= 1
					e--
				}
				mant &= 0x3FF
				f = (sign << 31) | ((e + 127 - 15) << 23) | (mant << 13)
			}
		} else if exp == 0x1F {
			f = (sign << 31) | 0x7F800000 | (mant << 13)
		} else {
			f = (sign << 31) | ((exp - 15 + 127) << 23) | (mant << 13)
		}
		half2floatLUT[h] = math.Float32frombits(f)
	}
}

// half2float converts IEEE 754 binary16 to float32 via lookup table
func half2float(h uint16) float32 {
	return half2floatLUT[h]
}
