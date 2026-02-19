package main

// quant.go — Quantized matrix operations and math utilities
//
// Supports F32, F16, Q4_0, Q5_0, Q8_0, Q4_K, Q6_K formats.
// All matmuls are parallelized across rows using goroutines.

import (
	"encoding/binary"
	"math"
	"runtime"
	"sync"
)

// Number of goroutines for parallel matmul
var numWorkers = runtime.NumCPU()

const q4BlockSize = 32
const q4BytesPerBlock = 18

// DequantQ4_0Block dequantizes a single Q4_0 block (32 values) into out
func DequantQ4_0Block(block []byte, out []float32) {
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))
	for j := 0; j < 16; j++ {
		b := block[2+j]
		v0 := int(b&0x0F) - 8
		v1 := int(b>>4) - 8
		out[j] = float32(v0) * d
		out[j+16] = float32(v1) * d
	}
}

// DequantQ4_0 dequantizes a full Q4_0 tensor into float32
func DequantQ4_0(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q4BlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q4BytesPerBlock
		DequantQ4_0Block(data[off:off+q4BytesPerBlock], out[i*q4BlockSize:])
	}
	return out
}

// MatMulQ4_0 computes out[rows] = W_q4[rows, cols] @ x[cols]
func MatMulQ4_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q4BlockSize
	bytesPerRow := blocksPerRow * q4BytesPerBlock

	if rows < numWorkers*4 {
		matMulQ4_0Range(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ4_0Range(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ4_0Range(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))
			xOff := b * q4BlockSize
			blockData := w[blockOff+2 : blockOff+q4BytesPerBlock]
			var dot float32
			for j := 0; j < 16; j++ {
				bv := blockData[j]
				v0 := float32(int(bv&0x0F) - 8)
				v1 := float32(int(bv>>4) - 8)
				dot += v0*x[xOff+j] + v1*x[xOff+j+16]
			}
			sum += dot * d
		}
		out[i] = sum
	}
}

// ============================================================
// Q8_0 dequantization (GGML type 8)
// ============================================================

const q8BlockSize = 32
const q8BytesPerBlock = 34

func DequantQ8_0Block(block []byte, out []float32) {
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))
	for j := 0; j < 32; j++ {
		out[j] = float32(int8(block[2+j])) * d
	}
}

func DequantQ8_0(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q8BlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q8BytesPerBlock
		DequantQ8_0Block(data[off:off+q8BytesPerBlock], out[i*q8BlockSize:])
	}
	return out
}

func MatMulQ8_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q8BlockSize
	bytesPerRow := blocksPerRow * q8BytesPerBlock

	if rows < numWorkers*4 {
		matMulQ8_0Range(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ8_0Range(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ8_0Range(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for i := start; i < end; i++ {
		rowOff := i * bytesPerRow
		sum := float32(0)
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q8BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))
			xOff := b * q8BlockSize
			var dot float32
			for j := 0; j < 32; j++ {
				dot += float32(int8(w[blockOff+2+j])) * x[xOff+j]
			}
			sum += dot * d
		}
		out[i] = sum
	}
}

// ============================================================
// Q6_K dequantization (GGML type 14)
// ============================================================

const q6kBlockSize = 256
const q6kBytesPerBlock = 210

func DequantQ6_K(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q6kBlockSize

	for i := 0; i < nblocks; i++ {
		blockOff := i * q6kBytesPerBlock
		ql := data[blockOff:]
		qh := data[blockOff+128:]
		scales := data[blockOff+192:]
		d := half2float(binary.LittleEndian.Uint16(data[blockOff+208 : blockOff+210]))

		outOff := i * q6kBlockSize

		for n128 := 0; n128 < 2; n128++ {
			qlP := ql[n128*64:]
			qhP := qh[n128*32:]
			scP := scales[n128*8:]
			yOff := outOff + n128*128

			for l := 0; l < 32; l++ {
				is := l / 16
				q1 := int(qlP[l]&0x0F) | (int(qhP[l]>>0)&3)<<4
				q2 := int(qlP[l+32]&0x0F) | (int(qhP[l]>>2)&3)<<4
				q3 := int(qlP[l]>>4) | (int(qhP[l]>>4)&3)<<4
				q4 := int(qlP[l+32]>>4) | (int(qhP[l]>>6)&3)<<4

				out[yOff+l+0] = d * float32(int8(scP[is+0])) * float32(q1-32)
				out[yOff+l+32] = d * float32(int8(scP[is+2])) * float32(q2-32)
				out[yOff+l+64] = d * float32(int8(scP[is+4])) * float32(q3-32)
				out[yOff+l+96] = d * float32(int8(scP[is+6])) * float32(q4-32)
			}
		}
	}
	return out
}

func MatMulQ6_K(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q6kBlockSize
	bytesPerRow := blocksPerRow * q6kBytesPerBlock

	if rows < numWorkers*4 {
		matMulQ6_KRange(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ6_KRange(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ6_KRange(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for r := start; r < end; r++ {
		rowOff := r * bytesPerRow
		sum := float32(0)
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q6kBytesPerBlock
			ql := w[blockOff:]
			qh := w[blockOff+128:]
			scales := w[blockOff+192:]
			d := half2float(binary.LittleEndian.Uint16(w[blockOff+208 : blockOff+210]))
			xOff := b * q6kBlockSize
			for n128 := 0; n128 < 2; n128++ {
				qlP := ql[n128*64:]
				qhP := qh[n128*32:]
				scP := scales[n128*8:]
				xBase := xOff + n128*128
				for l := 0; l < 32; l++ {
					is := l / 16
					q1 := int(qlP[l]&0x0F) | (int(qhP[l]>>0)&3)<<4
					q2 := int(qlP[l+32]&0x0F) | (int(qhP[l]>>2)&3)<<4
					q3 := int(qlP[l]>>4) | (int(qhP[l]>>4)&3)<<4
					q4 := int(qlP[l+32]>>4) | (int(qhP[l]>>6)&3)<<4

					s0 := d * float32(int8(scP[is+0]))
					s2 := d * float32(int8(scP[is+2]))
					s4 := d * float32(int8(scP[is+4]))
					s6 := d * float32(int8(scP[is+6]))

					sum += s0 * float32(q1-32) * x[xBase+l+0]
					sum += s2 * float32(q2-32) * x[xBase+l+32]
					sum += s4 * float32(q3-32) * x[xBase+l+64]
					sum += s6 * float32(q4-32) * x[xBase+l+96]
				}
			}
		}
		out[r] = sum
	}
}

// ============================================================
// Q4_K dequantization (GGML type 12)
// ============================================================

const q4kBlockSize = 256
const q4kBytesPerBlock = 144

func getScaleMinK4(j int, scales []byte) (sc, m uint8) {
	if j < 4 {
		sc = scales[j] & 63
		m = scales[j+4] & 63
	} else {
		sc = (scales[j+4] & 0x0F) | ((scales[j-4] >> 6) << 4)
		m = (scales[j+4] >> 4) | ((scales[j] >> 6) << 4)
	}
	return
}

func DequantQ4_KBlock(block []byte, out []float32) {
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))
	dmin := half2float(binary.LittleEndian.Uint16(block[2:4]))
	scales := block[4:16]
	qs := block[16:]

	is := 0
	outIdx := 0
	qIdx := 0
	for j := 0; j < q4kBlockSize; j += 64 {
		sc0, m0 := getScaleMinK4(is, scales)
		d1 := d * float32(sc0)
		m1 := dmin * float32(m0)
		sc1, m1v := getScaleMinK4(is+1, scales)
		d2 := d * float32(sc1)
		m2 := dmin * float32(m1v)

		for l := 0; l < 32; l++ {
			out[outIdx+l] = d1*float32(qs[qIdx+l]&0x0F) - m1
		}
		for l := 0; l < 32; l++ {
			out[outIdx+32+l] = d2*float32(qs[qIdx+l]>>4) - m2
		}
		qIdx += 32
		outIdx += 64
		is += 2
	}
}

func DequantQ4_K(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q4kBlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q4kBytesPerBlock
		DequantQ4_KBlock(data[off:off+q4kBytesPerBlock], out[i*q4kBlockSize:])
	}
	return out
}

func MatMulQ4_K(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q4kBlockSize
	bytesPerRow := blocksPerRow * q4kBytesPerBlock

	if rows < numWorkers*4 {
		matMulQ4_KRange(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ4_KRange(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ4_KRange(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for r := start; r < end; r++ {
		rowOff := r * bytesPerRow
		sum := float32(0)
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q4kBytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))
			dmin := half2float(binary.LittleEndian.Uint16(w[blockOff+2 : blockOff+4]))
			scales := w[blockOff+4 : blockOff+16]
			qs := w[blockOff+16:]
			xOff := b * q4kBlockSize
			is := 0
			qIdx := 0
			for j := 0; j < q4kBlockSize; j += 64 {
				sc0, m0 := getScaleMinK4(is, scales)
				d1 := d * float32(sc0)
				m1 := dmin * float32(m0)
				sc1, m1v := getScaleMinK4(is+1, scales)
				d2 := d * float32(sc1)
				m2 := dmin * float32(m1v)
				for l := 0; l < 32; l++ {
					sum += (d1*float32(qs[qIdx+l]&0x0F) - m1) * x[xOff+j+l]
				}
				for l := 0; l < 32; l++ {
					sum += (d2*float32(qs[qIdx+l]>>4) - m2) * x[xOff+j+32+l]
				}
				qIdx += 32
				is += 2
			}
		}
		out[r] = sum
	}
}

// ============================================================
// Q5_0 dequantization (GGML type 6)
// ============================================================

const q50BlockSize = 32
const q50BytesPerBlock = 22

func DequantQ5_0Block(block []byte, out []float32) {
	d := half2float(binary.LittleEndian.Uint16(block[0:2]))
	qh := binary.LittleEndian.Uint32(block[2:6])
	qs := block[6:22]

	for j := 0; j < 16; j++ {
		lo := int(qs[j] & 0x0F)
		hi := int(qs[j] >> 4)
		hbit0 := int((qh >> uint(j)) & 1)
		hbit1 := int((qh >> uint(j+16)) & 1)
		q0 := lo | (hbit0 << 4)
		q1 := hi | (hbit1 << 4)
		out[j] = float32(q0-16) * d
		out[j+16] = float32(q1-16) * d
	}
}

func DequantQ5_0(data []byte, n int) []float32 {
	out := make([]float32, n)
	nblocks := n / q50BlockSize
	for i := 0; i < nblocks; i++ {
		off := i * q50BytesPerBlock
		DequantQ5_0Block(data[off:off+q50BytesPerBlock], out[i*q50BlockSize:])
	}
	return out
}

func MatMulQ5_0(out []float32, w []byte, x []float32, rows, cols int) {
	blocksPerRow := cols / q50BlockSize
	bytesPerRow := blocksPerRow * q50BytesPerBlock

	if rows < numWorkers*4 {
		matMulQ5_0Range(out, w, x, 0, rows, blocksPerRow, bytesPerRow)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulQ5_0Range(out, w, x, s, e, blocksPerRow, bytesPerRow)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulQ5_0Range(out []float32, w []byte, x []float32, start, end, blocksPerRow, bytesPerRow int) {
	for r := start; r < end; r++ {
		rowOff := r * bytesPerRow
		sum := float32(0)
		for b := 0; b < blocksPerRow; b++ {
			blockOff := rowOff + b*q50BytesPerBlock
			d := half2float(binary.LittleEndian.Uint16(w[blockOff : blockOff+2]))
			qh := binary.LittleEndian.Uint32(w[blockOff+2 : blockOff+6])
			qs := w[blockOff+6:]
			xOff := b * q50BlockSize
			for j := 0; j < 16; j++ {
				lo := int(qs[j] & 0x0F)
				hi := int(qs[j] >> 4)
				hbit0 := int((qh >> uint(j)) & 1)
				hbit1 := int((qh >> uint(j+16)) & 1)
				q0 := lo | (hbit0 << 4)
				q1 := hi | (hbit1 << 4)
				sum += float32(q0-16) * d * x[xOff+j]
				sum += float32(q1-16) * d * x[xOff+j+16]
			}
		}
		out[r] = sum
	}
}

// ============================================================
// F32 and F16 matmul
// ============================================================

func MatMulF32(out []float32, w []float32, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		matMulF32Range(out, w, x, 0, rows, cols)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulF32Range(out, w, x, s, e, cols)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulF32Range(out []float32, w []float32, x []float32, start, end, cols int) {
	for i := start; i < end; i++ {
		sum := float32(0)
		off := i * cols
		for j := 0; j < cols; j++ {
			sum += w[off+j] * x[j]
		}
		out[i] = sum
	}
}

func MatMulF16(out []float32, w []byte, x []float32, rows, cols int) {
	if rows < numWorkers*4 {
		matMulF16Range(out, w, x, 0, rows, cols)
		return
	}

	var wg sync.WaitGroup
	chunkSize := (rows + numWorkers - 1) / numWorkers
	for worker := 0; worker < numWorkers; worker++ {
		start := worker * chunkSize
		end := start + chunkSize
		if end > rows {
			end = rows
		}
		if start >= end {
			break
		}
		wg.Add(1)
		go func(s, e int) {
			matMulF16Range(out, w, x, s, e, cols)
			wg.Done()
		}(start, end)
	}
	wg.Wait()
}

func matMulF16Range(out []float32, w []byte, x []float32, start, end, cols int) {
	for i := start; i < end; i++ {
		sum := float32(0)
		rowOff := i * cols * 2
		for j := 0; j < cols; j++ {
			wv := half2float(binary.LittleEndian.Uint16(w[rowOff+j*2 : rowOff+j*2+2]))
			sum += wv * x[j]
		}
		out[i] = sum
	}
}

// ============================================================
// Math utilities
// ============================================================

// RMSNorm applies RMS normalization in-place: x = x * w / rms(x)
func RMSNorm(x []float32, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		x[i] = x[i] * inv * w[i]
	}
}

// RMSNormBare applies RMS normalization in-place WITHOUT learnable weights.
// Used for QK-norm in nanollama (parameterless RMSNorm).
func RMSNormBare(x []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// RMSNormInto applies RMS normalization: out = norm(x) * w
func RMSNormInto(out, x, w []float32, eps float32) {
	n := len(x)
	var ss float64
	for i := 0; i < n; i++ {
		ss += float64(x[i]) * float64(x[i])
	}
	inv := float32(1.0 / math.Sqrt(ss/float64(n)+float64(eps)))
	for i := 0; i < n; i++ {
		out[i] = x[i] * inv * w[i]
	}
}

// Softmax computes softmax in-place over x[0:n]
func Softmax(x []float32, n int) {
	max := x[0]
	for i := 1; i < n; i++ {
		if x[i] > max {
			max = x[i]
		}
	}
	var sum float32
	for i := 0; i < n; i++ {
		x[i] = float32(math.Exp(float64(x[i] - max)))
		sum += x[i]
	}
	inv := float32(1.0) / sum
	for i := 0; i < n; i++ {
		x[i] *= inv
	}
}

// SiLU activation: x * sigmoid(x)
func SiLU(x float32) float32 {
	return x / (1.0 + float32(math.Exp(float64(-x))))
}
