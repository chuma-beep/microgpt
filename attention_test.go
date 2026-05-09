package main

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestAttentionHeadForward(t *testing.T) {
	headDim := 4
	T := 3

	// Simple test: uniform query, identity-like keys, uniform values
	q := []float64{1, 0, 0, 0}
	ks := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}
	vs := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{0, 0, 1, 0},
	}

	out, weights := attentionHead(q, ks, vs, headDim)

	// Check output shape
	if len(out) != headDim {
		t.Errorf("out len = %d, want %d", len(out), headDim)
	}
	if len(weights) != T {
		t.Errorf("weights len = %d, want %d", len(weights), T)
	}

	// Weights should sum to 1
	sum := 0.0
	for _, w := range weights {
		sum += w
		if w < 0 || w > 1 {
			t.Errorf("weight %f not in [0,1]", w)
		}
	}
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("weights sum = %f, want 1.0", sum)
	}
}

func TestAttentionHeadGradient(t *testing.T) {
	rng := rand.New(rand.NewPCG(123, 123))
	headDim := 4
	T := 3

	q := make([]float64, headDim)
	ks := make([][]float64, T)
	vs := make([][]float64, T)
	for d := range q {
		q[d] = rng.NormFloat64()
	}
	for i := 0; i < T; i++ {
		ks[i] = make([]float64, headDim)
		vs[i] = make([]float64, headDim)
		for d := 0; d < headDim; d++ {
			ks[i][d] = rng.NormFloat64()
			vs[i][d] = rng.NormFloat64()
		}
	}

	// Forward pass
	_, weights := attentionHead(q, ks, vs, headDim)

	// Loss = sum of output
	dout := make([]float64, headDim)
	for d := range dout {
		dout[d] = 1.0
	}

	// Analytical gradients
	dq, dks, dvs := gradAttentionHead(dout, q, ks, vs, weights, headDim)

	// Gradient check for query
	eps := 1e-5
	tol := 1e-3
	for d := 0; d < headDim; d++ {
		save := q[d]
		q[d] = save + eps
		outPlus, _ := attentionHead(q, ks, vs, headDim)
		lossPlus := sum(outPlus)

		q[d] = save - eps
		outMinus, _ := attentionHead(q, ks, vs, headDim)
		lossMinus := sum(outMinus)

		q[d] = save

		numerical := (lossPlus - lossMinus) / (2 * eps)
		relErr := relativeError(dq[d], numerical)
		if relErr > tol {
			t.Errorf("dq[%d] = %f, numerical = %f, relErr = %e", d, dq[d], numerical, relErr)
		}
	}

	// Gradient check for keys and values
	for i := 0; i < T; i++ {
		for d := 0; d < headDim; d++ {
			// Check dk
			save := ks[i][d]
			ks[i][d] = save + eps
			outPlus, _ := attentionHead(q, ks, vs, headDim)
			lossPlus := sum(outPlus)

			ks[i][d] = save - eps
			outMinus, _ := attentionHead(q, ks, vs, headDim)
			lossMinus := sum(outMinus)

			ks[i][d] = save

			numerical := (lossPlus - lossMinus) / (2 * eps)
			relErr := relativeError(dks[i][d], numerical)
			if relErr > tol {
				t.Errorf("dks[%d][%d] = %f, numerical = %f, relErr = %e", i, d, dks[i][d], numerical, relErr)
			}

			// Check dv
			save = vs[i][d]
			vs[i][d] = save + eps
			outPlus, _ = attentionHead(q, ks, vs, headDim)
			lossPlus = sum(outPlus)

			vs[i][d] = save - eps
			outMinus, _ = attentionHead(q, ks, vs, headDim)
			lossMinus = sum(outMinus)

			vs[i][d] = save

			numerical = (lossPlus - lossMinus) / (2 * eps)
			relErr = relativeError(dvs[i][d], numerical)
			if relErr > tol {
				t.Errorf("dvs[%d][%d] = %f, numerical = %f, relErr = %e", i, d, dvs[i][d], numerical, relErr)
			}
		}
	}
}

func relativeError(analytical, numerical float64) float64 {
	denom := math.Abs(analytical) + math.Abs(numerical) + 1e-10
	return math.Abs(analytical-numerical) / denom
}
