// matutil.go – all math primitives on []float64, manual matrix ops
package main

import "math"

func softmax(x []float64) []float64 {
	maxVal := x[0]
	for _, v := range x[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	exps := make([]float64, len(x))
	sum := 0.0
	for i, v := range x {
		exps[i] = math.Exp(v - maxVal)
		sum += exps[i]
	}
	out := make([]float64, len(x))
	for i, e := range exps {
		out[i] = e / sum
	}
	return out
}

func rmsnorm(x []float64) []float64 {
	ms := 0.0
	for _, v := range x {
		ms += v * v
	}
	ms /= float64(len(x))
	scale := 1.0 / math.Sqrt(ms+1e-5)
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = v * scale
	}
	return out
}

func relu(x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		if v > 0 {
			out[i] = v
		}
	}
	return out
}

func matVecMul(w []float64, x []float64, rows, cols int) []float64 {
	if len(x) != cols {
		panic("matVecMul: x length != cols")
	}
	out := make([]float64, rows)
	for i := 0; i < rows; i++ {
		sum := 0.0
		rowStart := i * cols
		for j := 0; j < cols; j++ {
			sum += w[rowStart+j] * x[j]
		}
		out[i] = sum
	}
	return out
}

func matVecMulTranspose(w []float64, x []float64, rows, cols int) []float64 {
	if len(x) != rows {
		panic("matVecMulTranspose: x length != rows")
	}
	out := make([]float64, cols)
	for j := 0; j < cols; j++ {
		sum := 0.0
		for i := 0; i < rows; i++ {
			sum += w[i*cols+j] * x[i]
		}
		out[j] = sum
	}
	return out
}

func vecAdd(a, b []float64) []float64 {
	if len(a) != len(b) {
		panic("vecAdd: length mismatch")
	}
	out := make([]float64, len(a))
	for i := range a {
		out[i] = a[i] + b[i]
	}
	return out
}

func vecScale(c float64, x []float64) []float64 {
	out := make([]float64, len(x))
	for i, v := range x {
		out[i] = c * v
	}
	return out
}

func outer(a, b []float64) (data []float64, rows, cols int) {
	rows = len(a)
	cols = len(b)
	data = make([]float64, rows*cols)
	for i := 0; i < rows; i++ {
		base := i * cols
		for j := 0; j < cols; j++ {
			data[base+j] = a[i] * b[j]
		}
	}
	return
}

// Gradient functions
func gradLinear(dout, w, x []float64, wRows, wCols int) (dw, dx []float64) {
	dw, _, _ = outer(dout, x)
	dx = matVecMulTranspose(w, dout, wRows, wCols)
	return
}

func gradRelu(dout, x []float64) []float64 {
	dx := make([]float64, len(x))
	for i := range x {
		if x[i] > 0 {
			dx[i] = dout[i]
		}
	}
	return dx
}

func gradRMSNorm(dout, x []float64) []float64 {
	n := float64(len(x))
	ms := 0.0
	for _, v := range x {
		ms += v * v
	}
	ms /= n
	invRms := 1.0 / math.Sqrt(ms+1e-5)
	sum := 0.0
	for i := range x {
		sum += dout[i] * x[i]
	}
	factor := sum * invRms * invRms * invRms / n
	dx := make([]float64, len(x))
	for i := range x {
		dx[i] = dout[i]*invRms - x[i]*factor
	}
	return dx
}
