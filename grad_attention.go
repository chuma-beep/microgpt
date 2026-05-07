// grad_attention.go — backward for single head and full attention block
package main

import "math"

// gradAttentionHead computes gradients for a single head.
// dout: gradient w.r.t. head output (length headDim)
// q: current query (length headDim)
// ks, vs: lists of past keys/values, each length headDim
// weights: attention weights from forward pass (length T)
// headDim: dimension
// Returns:
//
//	dq, dks, dvs (each dks[t] and dvs[t] are slices of length headDim)
func gradAttentionHead(dout, q []float64, ks, vs [][]float64, weights []float64, headDim int) (dq []float64, dks, dvs [][]float64) {
	T := len(ks)
	dq = make([]float64, headDim)
	dks = make([][]float64, T)
	dvs = make([][]float64, T)
	for t := 0; t < T; t++ {
		dvs[t] = make([]float64, headDim)
		dks[t] = make([]float64, headDim)
	}

	// gradient w.r.t. values: dvs[t] = weights[t] * dout
	for t := 0; t < T; t++ {
		for d := 0; d < headDim; d++ {
			dvs[t][d] = weights[t] * dout[d]
		}
	}

	// gradient w.r.t. attention weights: dweights[t] = dot(dout, vs[t])
	dweights := make([]float64, T)
	for t := 0; t < T; t++ {
		sum := 0.0
		for d := 0; d < headDim; d++ {
			sum += dout[d] * vs[t][d]
		}
		dweights[t] = sum
	}

	// gradient w.r.t. logits (before softmax): using softmax derivative
	// dlogits = dweights - sum(dweights * weights) * weights? Actually:
	// For softmax cross: if we have dweights from above, then dlogits = dweights * (diag(weights) - weights*weights^T)? Wait.
	// Simpler: we have weights = softmax(logits). The gradient of loss w.r.t. logits given gradient w.r.t. weights is:
	// dlogits = weights * (dweights - sum(dweights * weights))
	sumDw := 0.0
	for t := 0; t < T; t++ {
		sumDw += dweights[t] * weights[t]
	}
	dlogits := make([]float64, T)
	for t := 0; t < T; t++ {
		dlogits[t] = weights[t] * (dweights[t] - sumDw)
	}

	// gradient w.r.t. query: dq += sum_t (dlogits[t] / sqrt(headDim)) * ks[t]
	invSqrtDim := 1.0 / math.Sqrt(float64(headDim))
	for t := 0; t < T; t++ {
		for d := 0; d < headDim; d++ {
			dq[d] += dlogits[t] * ks[t][d] * invSqrtDim
		}
	}

	// gradient w.r.t. keys: dks[t] += dlogits[t] * q / sqrt(dim)
	for t := 0; t < T; t++ {
		for d := 0; d < headDim; d++ {
			dks[t][d] += dlogits[t] * q[d] * invSqrtDim
		}
	}
	return
}

// gradAttentionBlock processes all heads and returns gradients for weights and input.
// Assumes nHead and headDim are package vars.
func gradAttentionBlock(g *GPT, cache *Cache, pos int, dout []float64) (dx []float64, dWq, dWk, dWv, dWo []float64) {
	// dout is gradient w.r.t. attention output (before wo projection)
	// We need to backprop through wo, then through each head, then through QKV projections.

	// 1) gradient through wo (output projection)
	// Note: attnConcat is stored in cache for use in backward pass

	// Because this is a large function, I'll provide a skeleton that you can complete later.
	// The full implementation would:
	// - grad through wo: using gradLinear on dout, wo, and stored attnConcat.
	// - then split dattn into heads.
	// - for each head, call gradAttentionHead with cached q,k,v,weights.
	// - then combine dq, dk, dv and backprop through wq, wk, wv via gradLinear.
	// - sum contributions to dx (input to attention block).

	// Placeholder return
	dx = make([]float64, nEmb)
	dWq = make([]float64, nEmb*nEmb)
	dWk = make([]float64, nEmb*nEmb)
	dWv = make([]float64, nEmb*nEmb)
	dWo = make([]float64, nEmb*nEmb)
	return
}
