// cache.go – stores all activations for backward pass
package main

type Cache struct {
	Positions int
	Logits    [][]float64
	Targets   []int
	Tokens    []int // input tokens for embedding gradient

	// Pre‑attention block
	X    [][]float64 // after first RMSNorm (input to attention block)
	XRes [][]float64 // residual before attention (token+pos sum)

	// Attention internal
	Q, K, V     [][]float64 // projected vectors (full nEmb)
	AttnWeights [][]float64 // flattened across heads
	AttnConcat  [][]float64 // concatenated head outputs before WO projection

	// MLP block
	XResMlp [][]float64 // residual before MLP (output of attention block)
	MLPIn   [][]float64 // after RMSNorm before fc1
	MLPReLU [][]float64 // after ReLU
	MLPOut  [][]float64 // after fc2, before residual add
}
