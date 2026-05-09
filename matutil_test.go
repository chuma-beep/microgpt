package main

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestSoftmax(t *testing.T) {
	x := []float64{1, 2, 3, 4}
	out := softmax(x)

	// Check sum to 1
	sum := 0.0
	for _, v := range out {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax sum = %f, want 1.0", sum)
	}

	// Values should be in proportion to exp(x)
	// e^1=2.718, e^2=7.389, e^3=20.086, e^4=54.598, sum ~84.79
	// Even with max-subtraction, ratios should be exp-based
	for i := 1; i < len(out); i++ {
		if out[i] <= out[i-1] {
			t.Errorf("softmax not monotonic: out[%d]=%f <= out[%d]=%f", i, out[i], i-1, out[i-1])
		}
	}

	// Test numerical stability with large values
	large := []float64{1000, 1001, 1002}
	largeOut := softmax(large)
	for _, v := range largeOut {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Error("softmax produced NaN or Inf for large inputs")
		}
	}
	sum = 0.0
	for _, v := range largeOut {
		sum += v
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("softmax sum for large inputs = %f, want 1.0", sum)
	}
}

func TestSoftmaxSingleElement(t *testing.T) {
	out := softmax([]float64{5})
	if len(out) != 1 || out[0] != 1.0 {
		t.Errorf("softmax of single element = %v, want [1.0]", out)
	}
}

func TestRMSNorm(t *testing.T) {
	x := []float64{3, 4}
	out := rmsnorm(x, nil)

	// ms = (9+16)/2 = 12.5, scale = 1/sqrt(12.5+1e-5) ≈ 0.282843
	expected0 := 3 * 0.282843
	expected1 := 4 * 0.282843
	if math.Abs(out[0]-expected0) > 1e-5 {
		t.Errorf("rmsnorm[0] = %f, want ~%f", out[0], expected0)
	}
	if math.Abs(out[1]-expected1) > 1e-5 {
		t.Errorf("rmsnorm[1] = %f, want ~%f", out[1], expected1)
	}

	// RMS of output should be ~1
	ms := 0.0
	for _, v := range out {
		ms += v * v
	}
	ms /= float64(len(out))
	rms := math.Sqrt(ms)
	if math.Abs(rms-1.0) > 1e-5 {
		t.Errorf("RMS of normalized output = %f, want ~1.0", rms)
	}
}

func TestRMSNormWithGamma(t *testing.T) {
	x := []float64{3, 4}
	gamma := []float64{2.0, 0.5}
	out := rmsnorm(x, gamma)

	g := rmsnorm(x, nil)
	expected0 := g[0] * 2.0
	expected1 := g[1] * 0.5
	if math.Abs(out[0]-expected0) > 1e-8 {
		t.Errorf("rmsnorm with gamma[0] = %f, want ~%f", out[0], expected0)
	}
	if math.Abs(out[1]-expected1) > 1e-8 {
		t.Errorf("rmsnorm with gamma[1] = %f, want ~%f", out[1], expected1)
	}
}

func TestReLU(t *testing.T) {
	tests := []struct {
		input    float64
		expected float64
	}{
		{-2, 0},
		{-0.001, 0},
		{0, 0},
		{0.001, 0.001},
		{5, 5},
	}
	for _, tt := range tests {
		out := relu([]float64{tt.input})
		if math.Abs(out[0]-tt.expected) > 1e-10 {
			t.Errorf("relu(%f) = %f, want %f", tt.input, out[0], tt.expected)
		}
	}
}

func TestMatVecMul(t *testing.T) {
	// [[1,2,3],[4,5,6]] × [7,8,9]
	w := []float64{1, 2, 3, 4, 5, 6} // 2×3 matrix
	x := []float64{7, 8, 9}
	out := matVecMul(w, x, 2, 3)

	expected := []float64{1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9} // [50, 122]
	if len(out) != 2 || out[0] != expected[0] || out[1] != expected[1] {
		t.Errorf("matVecMul = %v, want [50, 122]", out)
	}
}

func TestMatVecMulTranspose(t *testing.T) {
	// [[1,2,3],[4,5,6]]^T × [7,8] = [[1,4],[2,5],[3,6]] × [7,8]
	w := []float64{1, 2, 3, 4, 5, 6} // 2×3 matrix
	x := []float64{7, 8}
	out := matVecMulTranspose(w, x, 2, 3)

	expected := []float64{1*7 + 4*8, 2*7 + 5*8, 3*7 + 6*8} // [39, 54, 69]
	if len(out) != 3 || out[0] != expected[0] || out[1] != expected[1] || out[2] != expected[2] {
		t.Errorf("matVecMulTranspose = %v, want [39, 54, 69]", out)
	}
}

func TestVecAdd(t *testing.T) {
	a := []float64{1, 2, 3}
	b := []float64{4, 5, 6}
	out := vecAdd(a, b)
	for i, v := range out {
		if v != a[i]+b[i] {
			t.Errorf("vecAdd[%d] = %f, want %f", i, v, a[i]+b[i])
		}
	}
}

func TestVecScale(t *testing.T) {
	x := []float64{1, 2, 3}
	out := vecScale(2.5, x)
	for i, v := range out {
		if v != 2.5*x[i] {
			t.Errorf("vecScale[%d] = %f, want %f", i, v, 2.5*x[i])
		}
	}
}

func TestOuter(t *testing.T) {
	a := []float64{1, 2}
	b := []float64{3, 4, 5}
	data, rows, cols := outer(a, b)

	if rows != 2 || cols != 3 {
		t.Errorf("outer dims = (%d,%d), want (2,3)", rows, cols)
	}

	expected := []float64{1 * 3, 1 * 4, 1 * 5, 2 * 3, 2 * 4, 2 * 5}
	if len(data) != len(expected) {
		t.Fatalf("outer len = %d, want %d", len(data), len(expected))
	}
	for i, v := range expected {
		if data[i] != v {
			t.Errorf("outer[%d] = %f, want %f", i, data[i], v)
		}
	}
}

func TestGradRelu(t *testing.T) {
	x := []float64{-1, 0, 1, 2}
	dout := []float64{3, 4, 5, 6}
	dx := gradRelu(dout, x)

	expected := []float64{0, 0, 5, 6}
	for i, v := range expected {
		if dx[i] != v {
			t.Errorf("gradRelu[%d] = %f, want %f", i, dx[i], v)
		}
	}
}

func TestGradRMSNorm(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	x := make([]float64, 8)
	for i := range x {
		x[i] = rng.NormFloat64()
	}

	// Gradient check via finite differences
	eps := 1e-5
	tol := 1e-4

	// Loss = sum of output (simple scalar loss for gradient check)
	// dout = ones
	dout := make([]float64, len(x))
	for i := range dout {
		dout[i] = 1.0
	}

	analytical := gradRMSNorm(dout, x)

	for i := range x {
		save := x[i]

		x[i] = save + eps
		yPlus := rmsnorm(x, nil)
		lossPlus := sum(yPlus)

		x[i] = save - eps
		yMinus := rmsnorm(x, nil)
		lossMinus := sum(yMinus)

		x[i] = save

		numerical := (lossPlus - lossMinus) / (2 * eps)
		if math.Abs(analytical[i]-numerical) > tol {
			t.Errorf("gradRMSNorm[%d] = %f, numerical = %f, diff = %e", i, analytical[i], numerical, analytical[i]-numerical)
		}
	}
}

func TestGradLinear(t *testing.T) {
	rng := rand.New(rand.NewPCG(42, 42))
	cols := 4
	rows := 3
	w := make([]float64, rows*cols)
	x := make([]float64, cols)
	for i := range w {
		w[i] = rng.NormFloat64()
	}
	for i := range x {
		x[i] = rng.NormFloat64()
	}

	// dout = ones
	dout := make([]float64, rows)
	for i := range dout {
		dout[i] = 1.0
	}

	dw, dx := gradLinear(dout, w, x, rows, cols)

	// Check dw via finite differences
	eps := 1e-5
	tol := 1e-4
	for i := range w {
		save := w[i]
		w[i] = save + eps
		yPlus := matVecMul(w, x, rows, cols)
		lossPlus := sum(yPlus)

		w[i] = save - eps
		yMinus := matVecMul(w, x, rows, cols)
		lossMinus := sum(yMinus)

		w[i] = save

		numerical := (lossPlus - lossMinus) / (2 * eps)
		if math.Abs(dw[i]-numerical) > tol {
			t.Errorf("dw[%d] = %f, numerical = %f, diff = %e", i, dw[i], numerical, dw[i]-numerical)
		}
	}

	// Check dx via finite differences
	for i := range x {
		save := x[i]
		x[i] = save + eps
		yPlus := matVecMul(w, x, rows, cols)
		lossPlus := sum(yPlus)

		x[i] = save - eps
		yMinus := matVecMul(w, x, rows, cols)
		lossMinus := sum(yMinus)

		x[i] = save

		numerical := (lossPlus - lossMinus) / (2 * eps)
		if math.Abs(dx[i]-numerical) > tol {
			t.Errorf("dx[%d] = %f, numerical = %f, diff = %e", i, dx[i], numerical, dx[i]-numerical)
		}
	}
}

func sum(s []float64) float64 {
	total := 0.0
	for _, v := range s {
		total += v
	}
	return total
}
