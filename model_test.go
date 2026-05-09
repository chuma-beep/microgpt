package main

import (
	"math"
	"os"
	"testing"
)

func namesForTest() []string {
	return []string{
		"alice", "bob", "charlie", "diana", "eve",
		"frank", "grace", "henry", "iris", "jack",
	}
}

func TestNewGPT(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	expectedParams := map[string][2]int{
		"wte":               {tok.VocabSize, nEmb},
		"wpe":               {blockSize, nEmb},
		"lm_head":           {tok.VocabSize, nEmb},
		"layer0.attn_wq":    {nEmb, nEmb},
		"layer0.attn_wk":    {nEmb, nEmb},
		"layer0.attn_wv":    {nEmb, nEmb},
		"layer0.attn_wo":    {nEmb, nEmb},
		"layer0.mlp_fc1":    {4 * nEmb, nEmb},
		"layer0.mlp_fc2":    {nEmb, 4 * nEmb},
		"layer0.rms1_gamma": {1, nEmb},
		"layer0.rms2_gamma": {1, nEmb},
	}

	if len(gpt.params) != 11 {
		t.Errorf("params count = %d, want 11", len(gpt.params))
	}
	if len(gpt.grads) != 11 {
		t.Errorf("grads count = %d, want 11", len(gpt.grads))
	}

	for name, dims := range expectedParams {
		m, ok := gpt.stateDict[name]
		if !ok {
			t.Errorf("missing parameter: %s", name)
			continue
		}
		if m.rows != dims[0] || m.cols != dims[1] {
			t.Errorf("%s dims = (%d,%d), want (%d,%d)", name, m.rows, m.cols, dims[0], dims[1])
		}
	}
}

func TestForwardSeqBasic(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	tokens := tok.Encode("alice")
	loss, cache := gpt.ForwardSeq(tokens)

	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		t.Errorf("ForwardSeq loss = %f", loss)
	}
	if loss <= 0 {
		t.Errorf("loss should be positive, got %f", loss)
	}
	if cache == nil {
		t.Fatal("cache is nil")
	}
	if cache.Positions == 0 {
		t.Error("cache.Positions is 0")
	}
	for pos := 0; pos < cache.Positions; pos++ {
		if len(cache.Logits[pos]) != tok.VocabSize {
			t.Errorf("logits[%d] len = %d, want %d", pos, len(cache.Logits[pos]), tok.VocabSize)
		}
	}
}

func TestForwardSeqBackwardNoPanic(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	tokens := tok.Encode("alice")
	_, cache := gpt.ForwardSeq(tokens)

	// Backward should not panic
	gpt.Backward(cache)

	// Verify some gradients are non-zero
	totalGrad := 0.0
	for _, gr := range gpt.grads {
		for _, v := range gr.data {
			totalGrad += math.Abs(v)
		}
	}
	if totalGrad == 0 {
		t.Error("all gradients are zero after backward")
	}
}

func TestDirectionalGradientCheck(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	tokens := tok.Encode("alice")
	eps := 1e-5

	// Step 1: Forward + Backward to get analytical gradients
	lossBase, cache := gpt.ForwardSeq(tokens)

	// Zero all gradients first
	for _, gr := range gpt.grads {
		for i := range gr.data {
			gr.data[i] = 0
		}
	}

	gpt.Backward(cache)

	// Flatten analytical gradients
	analyticalGrads := make([]float64, 0)
	for _, gr := range gpt.grads {
		analyticalGrads = append(analyticalGrads, gr.data...)
	}

	// Check analytical gradients are not all zero
	gradNorm := math.Sqrt(dot(analyticalGrads, analyticalGrads))
	if gradNorm < 1e-10 {
		t.Fatal("analytical gradient norm is zero")
	}
	t.Logf("gradient norm: %f", gradNorm)

	// Step 2: Directional gradient check with normalized direction
	// Use the analytical gradient direction itself for checking
	direction := make([]float64, len(analyticalGrads))
	copy(direction, analyticalGrads)
	dirNorm := math.Sqrt(dot(direction, direction))
	for i := range direction {
		direction[i] /= dirNorm
	}

	// Step 3: Save original params, compute numerical derivative
	originalFlat := make([]float64, len(direction))
	{
		idx := 0
		for _, m := range gpt.params {
			copy(originalFlat[idx:idx+len(m.data)], m.data)
			idx += len(m.data)
		}
	}

	// Perturb forward: params + eps * direction
	idx := 0
	for _, m := range gpt.params {
		for i := range m.data {
			m.data[i] = originalFlat[idx+i] + eps*direction[idx+i]
		}
		idx += len(m.data)
	}
	lossPlus, _ := gpt.ForwardSeq(tokens)

	// Perturb backward: params - eps * direction
	idx = 0
	for _, m := range gpt.params {
		for i := range m.data {
			m.data[i] = originalFlat[idx+i] - eps*direction[idx+i]
		}
		idx += len(m.data)
	}
	lossMinus, _ := gpt.ForwardSeq(tokens)

	// Restore original params
	idx = 0
	for _, m := range gpt.params {
		copy(m.data, originalFlat[idx:idx+len(m.data)])
		idx += len(m.data)
	}

	numDerivative := (lossPlus - lossMinus) / (2 * eps)
	analyticalDerivative := dot(analyticalGrads, direction)

	t.Logf("loss base: %.6f, plus: %.6f, minus: %.6f", lossBase, lossPlus, lossMinus)
	t.Logf("analytical directional: %f, numerical directional: %f", analyticalDerivative, numDerivative)

	// Verify signs match (gradients point in same direction)
	if analyticalDerivative*numDerivative <= 0 {
		t.Errorf("directional gradients have different signs: analytical=%f, numerical=%f", analyticalDerivative, numDerivative)
	}

	// Verify magnitudes are within reasonable ratio (within factor of 5)
	if math.Abs(analyticalDerivative) > 1e-8 && math.Abs(numDerivative) > 1e-8 {
		ratio := analyticalDerivative / numDerivative
		if ratio < 0.2 || ratio > 5.0 {
			t.Errorf("gradient magnitude ratio out of range: analytical/num = %f", ratio)
		}
	}
}

func TestGradientAllParamsNonZero(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	tokens := tok.Encode("alice")
	_, cache := gpt.ForwardSeq(tokens)

	for _, gr := range gpt.grads {
		for i := range gr.data {
			gr.data[i] = 0
		}
	}

	gpt.Backward(cache)

	zeroCount := 0
	totalCount := 0
	for name, gr := range gpt.gradMap {
		nonZero := false
		for _, v := range gr.data {
			if v != 0 {
				nonZero = true
				break
			}
		}
		totalCount++
		if !nonZero {
			zeroCount++
			t.Logf("WARNING: gradient for %s is all zeros", name)
		}
	}
	if zeroCount > 0 {
		t.Errorf("%d/%d parameter matrices have all-zero gradients", zeroCount, totalCount)
	}
}

func dot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func TestWeightSaveLoadRoundtrip(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)

	// Collect original parameter values
	originals := make([][]float64, len(gpt.params))
	for i, m := range gpt.params {
		originals[i] = make([]float64, len(m.data))
		copy(originals[i], m.data)
	}

	// Save to temp file
	tmpPath := "test_weights.bin"
	err := saveWeights(tmpPath, gpt.params)
	if err != nil {
		t.Fatalf("saveWeights failed: %v", err)
	}
	defer os.Remove(tmpPath)

	// Corrupt params
	for _, m := range gpt.params {
		for i := range m.data {
			m.data[i] = 0
		}
	}

	// Load back
	err = loadWeights(tmpPath, gpt.params)
	if err != nil {
		t.Fatalf("loadWeights failed: %v", err)
	}

	// Verify params match
	for i, m := range gpt.params {
		for j, v := range m.data {
			if v != originals[i][j] {
				t.Errorf("param[%d][%d] = %f, want %f after roundtrip", i, j, v, originals[i][j])
				return
			}
		}
	}
}

func TestTrainingConvergence(t *testing.T) {
	tok := NewTokenizer(namesForTest())
	gpt := NewGPT(tok)
	trainer := NewTrainer(gpt, tok, namesForTest())

	losses := make([]float64, 0, 200)
	for step := 0; step < 200; step++ {
		loss := trainer.Step()
		losses = append(losses, loss)
	}

	// Loss should decrease overall
	initialAvg := mean(losses[:20])
	finalAvg := mean(losses[180:])

	t.Logf("initial avg loss: %f, final avg loss: %f", initialAvg, finalAvg)

	if finalAvg >= initialAvg {
		t.Errorf("loss did not decrease: initial = %f, final = %f", initialAvg, finalAvg)
	}
}

func mean(s []float64) float64 {
	if len(s) == 0 {
		return 0
	}
	total := 0.0
	for _, v := range s {
		total += v
	}
	return total / float64(len(s))
}
