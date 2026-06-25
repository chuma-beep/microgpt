// cache.go – stores all activations for backward pass
package main

type Cache struct {
	Positions int
	Logits    [][]float64
	Probs     [][]float64 // softmax(logits) at each position
	Targets   []int
	Tokens    []int // input tokens for embedding gradient

	// Pre‑attention block
	X    [][]float64 // after first RMSNorm (input to attention block)
	XRes [][]float64 // residual before attention (token+pos sum)

	// Attention internal
	Q, K, V     [][]float64 // projected vectors (full nEmb)
	AttnScores  [][]float64 // pre-softmax QK^T/sqrt(d) per position, flattened across heads
	AttnWeights [][]float64 // flattened across heads
	AttnConcat  [][]float64 // concatenated head outputs before WO projection
	AttnProj    [][]float64 // WO projection output before residual add

	// MLP block
	XResMlp    [][]float64 // residual before MLP (output of attention block)
	MLPIn      [][]float64 // after RMSNorm before fc1
	MLPPreReLU [][]float64 // fc1 output before ReLU
	MLPReLU    [][]float64 // after ReLU
	MLPOut     [][]float64 // after fc2, before residual add

	// Final hidden state (after MLP residual, before lm_head)
	FinalX [][]float64
}
