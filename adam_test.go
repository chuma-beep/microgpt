package main

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestNewAdam(t *testing.T) {
	a := NewAdam(100)
	if a.t != 0 {
		t.Errorf("Adam.t = %d, want 0", a.t)
	}
	if len(a.m) != 100 || len(a.v) != 100 {
		t.Errorf("Adam m/v lengths = %d/%d, want 100", len(a.m), len(a.v))
	}
	if a.beta1 != 0.9 || a.beta2 != 0.999 {
		t.Errorf("Adam beta1/beta2 = %f/%f, want 0.9/0.999", a.beta1, a.beta2)
	}
}

func TestAdamUpdate(t *testing.T) {
	params := []float64{0.5, -0.3, 1.0}
	grads := []float64{0.1, -0.2, 0.0}
	lr := 0.01

	a := NewAdam(len(params))
	original := make([]float64, len(params))
	copy(original, params)

	a.Update(params, grads, lr)

	// Params with non-zero grad should change
	if params[0] == original[0] {
		t.Error("param[0] did not change with non-zero gradient")
	}

	// Param with zero grad should NOT change
	if params[2] != original[2] {
		t.Error("param[2] changed with zero gradient")
	}

	// Step counter should increment
	if a.t != 1 {
		t.Errorf("Adam.t = %d, want 1", a.t)
	}

	// Momentum and velocity should be updated
	if a.m[0] != (1-a.beta1)*grads[0] {
		t.Errorf("Adam.m[0] = %f, want %f (first step: (1-beta1)*grad)", a.m[0], (1-a.beta1)*grads[0])
	}
	if a.v[0] != (1-a.beta2)*grads[0]*grads[0] {
		t.Errorf("Adam.v[0] = %f, want %f", a.v[0], (1-a.beta2)*grads[0]*grads[0])
	}
}

func TestAdamUpdateDirection(t *testing.T) {
	// Positive gradient should decrease parameter
	a := NewAdam(1)
	params := []float64{1.0}
	grads := []float64{1.0}
	a.Update(params, grads, 0.1)
	if params[0] >= 1.0 {
		t.Error("positive gradient should decrease parameter")
	}

	// Negative gradient should increase parameter
	a2 := NewAdam(1)
	params2 := []float64{1.0}
	grads2 := []float64{-1.0}
	a2.Update(params2, grads2, 0.1)
	if params2[0] <= 1.0 {
		t.Error("negative gradient should increase parameter")
	}
}

func TestAdamMultipleSteps(t *testing.T) {
	rng := rand.New(rand.NewPCG(99, 99))
	n := 10
	params := make([]float64, n)
	for i := range params {
		params[i] = rng.NormFloat64()
	}
	original := make([]float64, n)
	copy(original, params)

	grads := make([]float64, n)
	for i := range grads {
		grads[i] = rng.NormFloat64()
	}

	a := NewAdam(n)
	for step := 0; step < 100; step++ {
		a.Update(params, grads, 0.001)
	}

	// After many steps, params should have moved meaningfully
	for i := range params {
		if math.Abs(params[i]-original[i]) < 1e-10 {
			t.Errorf("param[%d] did not change after 100 steps", i)
		}
	}
}
