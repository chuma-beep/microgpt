// attention.go – single head forward attention
package main

import "math"

func attentionHead(q []float64, ks, vs [][]float64, headDim int) (out []float64, weights []float64) {
	T := len(ks)
	logits := make([]float64, T)
	for t := 0; t < T; t++ {
		dot := 0.0
		for d := 0; d < headDim; d++ {
			dot += q[d] * ks[t][d]
		}
		logits[t] = dot / math.Sqrt(float64(headDim))
	}
	weights = softmax(logits)
	out = make([]float64, headDim)
	for d := 0; d < headDim; d++ {
		sum := 0.0
		for t := 0; t < T; t++ {
			sum += weights[t] * vs[t][d]
		}
		out[d] = sum
	}
	return
}
