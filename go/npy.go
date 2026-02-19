package main

// npy.go — NPY/NPZ file reading utilities
//
// Reads numpy .npy files (version 1 and 2) for gamma essence loading.
// Supports int32, float16, float32 dtypes.

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"strings"
)

// readNpyHeader reads and returns the npy header string
func readNpyHeader(r io.Reader) (string, error) {
	magic := make([]byte, 6)
	if _, err := io.ReadFull(r, magic); err != nil {
		return "", fmt.Errorf("read magic: %w", err)
	}
	if magic[0] != 0x93 || string(magic[1:6]) != "NUMPY" {
		return "", fmt.Errorf("not a npy file")
	}

	ver := make([]byte, 2)
	if _, err := io.ReadFull(r, ver); err != nil {
		return "", fmt.Errorf("read version: %w", err)
	}

	var headerLen int
	if ver[0] == 1 {
		hl := make([]byte, 2)
		if _, err := io.ReadFull(r, hl); err != nil {
			return "", fmt.Errorf("read header len: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint16(hl))
	} else {
		hl := make([]byte, 4)
		if _, err := io.ReadFull(r, hl); err != nil {
			return "", fmt.Errorf("read header len v2: %w", err)
		}
		headerLen = int(binary.LittleEndian.Uint32(hl))
	}

	header := make([]byte, headerLen)
	if _, err := io.ReadFull(r, header); err != nil {
		return "", fmt.Errorf("read header: %w", err)
	}
	return string(header), nil
}

// readNpyFloat reads a numpy .npy file and returns float32 data + 2D shape
// Supports float16 and float32 dtypes
func readNpyFloat(r io.Reader) ([]float32, [2]int, error) {
	hstr, err := readNpyHeader(r)
	if err != nil {
		return nil, [2]int{}, err
	}

	isFloat16 := strings.Contains(hstr, "'<f2'") || strings.Contains(hstr, "float16")
	isFloat32 := strings.Contains(hstr, "'<f4'") || strings.Contains(hstr, "float32")
	if !isFloat16 && !isFloat32 {
		return nil, [2]int{}, fmt.Errorf("unsupported dtype in header: %s", hstr)
	}

	shape := parseShape(hstr)
	if shape[0] == 0 || shape[1] == 0 {
		return nil, [2]int{}, fmt.Errorf("could not parse shape from header: %s", hstr)
	}

	totalElements := shape[0] * shape[1]

	var data []float32
	if isFloat16 {
		raw := make([]byte, totalElements*2)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, [2]int{}, fmt.Errorf("read float16 data: %w", err)
		}
		data = make([]float32, totalElements)
		for i := 0; i < totalElements; i++ {
			h := uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
			data[i] = half2float(h)
		}
	} else {
		raw := make([]byte, totalElements*4)
		if _, err := io.ReadFull(r, raw); err != nil {
			return nil, [2]int{}, fmt.Errorf("read float32 data: %w", err)
		}
		data = make([]float32, totalElements)
		for i := 0; i < totalElements; i++ {
			data[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[i*4:]))
		}
	}

	return data, shape, nil
}

// readNpyF16Raw reads a 2D float16 npy file and returns raw uint16 data (NO conversion to f32)
func readNpyF16Raw(r io.Reader) ([]uint16, [2]int, error) {
	hstr, err := readNpyHeader(r)
	if err != nil {
		return nil, [2]int{}, err
	}

	if !strings.Contains(hstr, "'<f2'") && !strings.Contains(hstr, "float16") {
		return nil, [2]int{}, fmt.Errorf("expected float16, got header: %s", hstr)
	}

	shape := parseShape(hstr)
	if shape[0] == 0 || shape[1] == 0 {
		return nil, [2]int{}, fmt.Errorf("could not parse shape from header: %s", hstr)
	}

	total := shape[0] * shape[1]
	raw := make([]byte, total*2)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, [2]int{}, fmt.Errorf("read f16 data: %w", err)
	}

	data := make([]uint16, total)
	for i := 0; i < total; i++ {
		data[i] = uint16(raw[i*2]) | uint16(raw[i*2+1])<<8
	}
	return data, shape, nil
}

// readNpyInt32 reads a 1D int32 numpy array
func readNpyInt32(r io.Reader) ([]int32, error) {
	header, err := readNpyHeader(r)
	if err != nil {
		return nil, err
	}

	if !strings.Contains(header, "'<i4'") && !strings.Contains(header, "int32") {
		return nil, fmt.Errorf("expected int32 dtype, got header: %s", header)
	}

	shape := parseShapeAny(header)
	if len(shape) == 0 {
		return nil, fmt.Errorf("could not parse shape from header: %s", header)
	}

	total := 1
	for _, s := range shape {
		total *= s
	}

	raw := make([]byte, total*4)
	if _, err := io.ReadFull(r, raw); err != nil {
		return nil, fmt.Errorf("read int32 data: %w", err)
	}

	data := make([]int32, total)
	for i := 0; i < total; i++ {
		data[i] = int32(binary.LittleEndian.Uint32(raw[i*4:]))
	}
	return data, nil
}

// parseShapeAny extracts shape tuple from npy header, supports 1D and 2D
func parseShapeAny(header string) []int {
	idx := strings.Index(header, "shape")
	if idx < 0 {
		return nil
	}

	start := strings.Index(header[idx:], "(")
	if start < 0 {
		return nil
	}
	start += idx + 1

	end := strings.Index(header[start:], ")")
	if end < 0 {
		return nil
	}

	shapeStr := strings.TrimSpace(header[start : start+end])
	if shapeStr == "" {
		return []int{1}
	}

	parts := strings.Split(shapeStr, ",")
	var shape []int
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		var v int
		fmt.Sscanf(p, "%d", &v)
		if v > 0 {
			shape = append(shape, v)
		}
	}
	return shape
}

// parseShape extracts (rows, cols) from npy header string (2D only)
func parseShape(header string) [2]int {
	s := parseShapeAny(header)
	if len(s) >= 2 {
		return [2]int{s[0], s[1]}
	}
	return [2]int{}
}
