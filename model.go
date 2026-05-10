// model.go – GPT struct, parameter init, forward, backward, generation
package main

import (
	"math"
	"math/rand/v2"
)

var (
	nEmb      = 16
	blockSize = 16
	nHead     = 4
	headDim   = nEmb / nHead // 4
	nLayer    = 1
)

type matrix struct {
	data []float64
	rows int
	cols int
}

func newMatrix(rows, cols int, std float64) *matrix {
	m := &matrix{
		data: make([]float64, rows*cols),
		rows: rows,
		cols: cols,
	}
	for i := range m.data {
		m.data[i] = rand.NormFloat64() * std
	}
	return m
}

func (m *matrix) row(idx int) []float64 {
	if idx < 0 || idx >= m.rows {
		panic("row index out of range")
	}
	start := idx * m.cols
	end := start + m.cols
	out := make([]float64, m.cols)
	copy(out, m.data[start:end])
	return out
}

func (m *matrix) setRow(idx int, vals []float64) {
	if len(vals) != m.cols {
		panic("row length mismatch")
	}
	start := idx * m.cols
	copy(m.data[start:start+m.cols], vals)
}

type GPT struct {
	stateDict map[string]*matrix
	params    []*matrix
	grads     []*matrix
	gradMap   map[string]*matrix // for easy access by name
	tok       *Tokenizer
}

func NewGPT(tok *Tokenizer) *GPT {
	g := &GPT{
		stateDict: make(map[string]*matrix),
		gradMap:   make(map[string]*matrix),
		tok:       tok,
	}
	addParam := func(name string, m *matrix) {
		g.stateDict[name] = m
		g.params = append(g.params, m)
		gradM := &matrix{data: make([]float64, m.rows*m.cols), rows: m.rows, cols: m.cols}
		g.grads = append(g.grads, gradM)
		g.gradMap[name] = gradM
	}
	vocabSize := tok.VocabSize
	addParam("wte", newMatrix(vocabSize, nEmb, 0.08))
	addParam("wpe", newMatrix(blockSize, nEmb, 0.08))
	addParam("lm_head", newMatrix(vocabSize, nEmb, 0.08))
	addParam("layer0.attn_wq", newMatrix(nEmb, nEmb, 0.08))
	addParam("layer0.attn_wk", newMatrix(nEmb, nEmb, 0.08))
	addParam("layer0.attn_wv", newMatrix(nEmb, nEmb, 0.08))
	addParam("layer0.attn_wo", newMatrix(nEmb, nEmb, 0.08))
	addParam("layer0.mlp_fc1", newMatrix(4*nEmb, nEmb, 0.08))
	addParam("layer0.mlp_fc2", newMatrix(nEmb, 4*nEmb, 0.08))
	addParam("layer0.rms1_gamma", newMatrix(1, nEmb, 0.0))
	addParam("layer0.rms2_gamma", newMatrix(1, nEmb, 0.0))
	// Initialize gamma to 1.0
	for i := range g.stateDict["layer0.rms1_gamma"].data {
		g.stateDict["layer0.rms1_gamma"].data[i] = 1.0
	}
	for i := range g.stateDict["layer0.rms2_gamma"].data {
		g.stateDict["layer0.rms2_gamma"].data[i] = 1.0
	}
	return g
}

func (g *GPT) ForwardSeq(tokens []int) (float64, *Cache) {
	n := min(blockSize, len(tokens)-1)
	if n < 1 {
		panic("sequence too short")
	}

	cache := &Cache{
		Positions:   n,
		Logits:      make([][]float64, n),
		Targets:     make([]int, n),
		Tokens:      make([]int, n),
		X:           make([][]float64, n),
		XRes:        make([][]float64, n),
		Q:           make([][]float64, n),
		K:           make([][]float64, n),
		V:           make([][]float64, n),
		AttnWeights: make([][]float64, n),
		AttnConcat:  make([][]float64, n),
		XResMlp:     make([][]float64, n),
		MLPIn:       make([][]float64, n),
		MLPReLU:     make([][]float64, n),
		MLPOut:      make([][]float64, n),
		FinalX:      make([][]float64, n),
	}
	wte := g.stateDict["wte"]
	wpe := g.stateDict["wpe"]
	lmHead := g.stateDict["lm_head"]
	rms1Gamma := g.stateDict["layer0.rms1_gamma"].data
	rms2Gamma := g.stateDict["layer0.rms2_gamma"].data

	keysCache := make([][][]float64, nLayer) // layer 0 only
	valuesCache := make([][][]float64, nLayer)

	totalLoss := 0.0
	for pos := 0; pos < n; pos++ {
		tokenID := tokens[pos]
		targetID := tokens[pos+1]
		tokEmb := wte.row(tokenID)
		posEmb := wpe.row(pos)
		x := vecAdd(tokEmb, posEmb)
		cache.XRes[pos] = append([]float64(nil), x...)
		x = rmsnorm(x, rms1Gamma)
		cache.X[pos] = append([]float64(nil), x...)

		// Attention block
		xResAttn := append([]float64(nil), x...)
		x = rmsnorm(x, rms1Gamma)

		wq := g.stateDict["layer0.attn_wq"]
		wk := g.stateDict["layer0.attn_wk"]
		wv := g.stateDict["layer0.attn_wv"]
		q := matVecMul(wq.data, x, wq.rows, wq.cols)
		k := matVecMul(wk.data, x, wk.rows, wk.cols)
		v := matVecMul(wv.data, x, wv.rows, wv.cols)
		cache.Q[pos] = append([]float64(nil), q...)
		cache.K[pos] = append([]float64(nil), k...)
		cache.V[pos] = append([]float64(nil), v...)

		keysCache[0] = append(keysCache[0], k)
		valuesCache[0] = append(valuesCache[0], v)

		attnOut := make([]float64, nEmb)
		allWeights := make([]float64, 0, nHead*len(keysCache[0]))
		for h := 0; h < nHead; h++ {
			hs := h * headDim
			he := hs + headDim
			qHead := q[hs:he]
			kHeads := make([][]float64, len(keysCache[0]))
			vHeads := make([][]float64, len(valuesCache[0]))
			for t := 0; t < len(keysCache[0]); t++ {
				kHeads[t] = keysCache[0][t][hs:he]
				vHeads[t] = valuesCache[0][t][hs:he]
			}
			headOut, wts := attentionHead(qHead, kHeads, vHeads, headDim)
			for j := 0; j < headDim; j++ {
				attnOut[hs+j] = headOut[j]
			}
			allWeights = append(allWeights, wts...)
		}
		cache.AttnWeights[pos] = allWeights
		cache.AttnConcat[pos] = append([]float64(nil), attnOut...)

		wo := g.stateDict["layer0.attn_wo"]
		attnProj := matVecMul(wo.data, attnOut, wo.rows, wo.cols)
		x = vecAdd(attnProj, xResAttn)

		// MLP block
		cache.XResMlp[pos] = append([]float64(nil), x...)
		x = rmsnorm(x, rms2Gamma)
		cache.MLPIn[pos] = append([]float64(nil), x...)
		fc1 := g.stateDict["layer0.mlp_fc1"]
		fc2 := g.stateDict["layer0.mlp_fc2"]
		x = matVecMul(fc1.data, x, fc1.rows, fc1.cols)
		x = relu(x)
		cache.MLPReLU[pos] = append([]float64(nil), x...)
		x = matVecMul(fc2.data, x, fc2.rows, fc2.cols)
		cache.MLPOut[pos] = append([]float64(nil), x...)
		x = vecAdd(x, cache.XResMlp[pos])
		cache.FinalX[pos] = append([]float64(nil), x...)

		logits := matVecMul(lmHead.data, x, lmHead.rows, lmHead.cols)
		cache.Logits[pos] = logits
		cache.Targets[pos] = targetID
		cache.Tokens[pos] = tokenID
		probs := softmax(logits)
		totalLoss += -math.Log(probs[targetID])
	}
	return totalLoss / float64(n), cache
}

func (g *GPT) Backward(cache *Cache) {
	n := cache.Positions
	vocabSize := g.tok.VocabSize

	dkAccum := make([][]float64, n)
	dvAccum := make([][]float64, n)
	dxFromAttn := make([][]float64, n)
	for i := 0; i < n; i++ {
		dkAccum[i] = make([]float64, nEmb)
		dvAccum[i] = make([]float64, nEmb)
		dxFromAttn[i] = make([]float64, nEmb)
	}

	dGamma1 := make([]float64, nEmb)
	dGamma2 := make([]float64, nEmb)

	for pos := n - 1; pos >= 0; pos-- {
		logits := cache.Logits[pos]
		target := cache.Targets[pos]
		probs := softmax(logits)

		dlogits := make([]float64, vocabSize)
		for i := range dlogits {
			dlogits[i] = probs[i]
		}
		dlogits[target] -= 1.0
		scale := 1.0 / float64(n)
		for i := range dlogits {
			dlogits[i] *= scale
		}

		lmHead := g.stateDict["lm_head"]
		dlmHeadWeight, _, _ := outer(dlogits, cache.FinalX[pos])
		addGrad(g.gradMap["lm_head"], dlmHeadWeight)
		dxAfterResidual := matVecMulTranspose(lmHead.data, dlogits, lmHead.rows, lmHead.cols)

		dx := dxAfterResidual

		fc2 := g.stateDict["layer0.mlp_fc2"]
		dfc2Weight, dfc2Input := gradLinear(dx, fc2.data, cache.MLPReLU[pos], fc2.rows, fc2.cols)
		addGrad(g.gradMap["layer0.mlp_fc2"], dfc2Weight)

		drelu := gradRelu(dfc2Input, cache.MLPReLU[pos])

		fc1 := g.stateDict["layer0.mlp_fc1"]
		dfc1Weight, dfc1Input := gradLinear(drelu, fc1.data, cache.MLPIn[pos], fc1.rows, fc1.cols)
		addGrad(g.gradMap["layer0.mlp_fc1"], dfc1Weight)

		dXNorm := dfc1Input
		dXResMlp := make([]float64, nEmb)
		for i := range dXResMlp {
			dXResMlp[i] = dXNorm[i]
		}
		for i := range dXResMlp {
			dXResMlp[i] += dx[i]
		}

		xBeforeMlpNorm := cache.XResMlp[pos]
		dXAfterMlpNorm := gradRMSNorm(elemMul(dXResMlp, g.stateDict["layer0.rms2_gamma"].data), xBeforeMlpNorm)
		addGradRMSNormGamma(dGamma2, dXResMlp, xBeforeMlpNorm)

		attnOut := cache.AttnConcat[pos]
		wo := g.stateDict["layer0.attn_wo"]
		dwoWeight, dAttnOut := gradLinear(dXAfterMlpNorm, wo.data, attnOut, wo.rows, wo.cols)
		addGrad(g.gradMap["layer0.attn_wo"], dwoWeight)

		// Per-position incremental dk/dv for this attention computation
		dkPerPos := make([][]float64, pos+1)
		dvPerPos := make([][]float64, pos+1)
		for t := 0; t <= pos; t++ {
			dkPerPos[t] = make([]float64, nEmb)
			dvPerPos[t] = make([]float64, nEmb)
		}

		dqFull := make([]float64, nEmb)
		for h := 0; h < nHead; h++ {
			hs := h * headDim
			he := hs + headDim
			dAttnOutHead := dAttnOut[hs:he]

			qHead := cache.Q[pos][hs:he]
			kHeads := make([][]float64, pos+1)
			vHeads := make([][]float64, pos+1)
			for t := 0; t <= pos; t++ {
				kHeads[t] = cache.K[t][hs:he]
				vHeads[t] = cache.V[t][hs:he]
			}

			weightStart := h * (pos + 1)
			attnWeights := cache.AttnWeights[pos][weightStart : weightStart+pos+1]

			dqHead, dkHeads, dvHeads := gradAttentionHead(dAttnOutHead, qHead, kHeads, vHeads, attnWeights, headDim)

			for j := 0; j < headDim; j++ {
				dqFull[hs+j] += dqHead[j]
			}
			for t := 0; t <= pos; t++ {
				for j := 0; j < headDim; j++ {
					dkAccum[t][hs+j] += dkHeads[t][j]
					dvAccum[t][hs+j] += dvHeads[t][j]
					dkPerPos[t][hs+j] += dkHeads[t][j]
					dvPerPos[t][hs+j] += dvHeads[t][j]
				}
			}
		}

		wq := g.stateDict["layer0.attn_wq"]
		wk := g.stateDict["layer0.attn_wk"]
		wv := g.stateDict["layer0.attn_wv"]

		// WQ gradient
		xNorm := cache.X[pos]
		dwqWeight, _, _ := outer(dqFull, xNorm)
		addGrad(g.gradMap["layer0.attn_wq"], dwqWeight)

		// WK/WV gradients from accumulated per-position dk/dv
		dwkWeight, _, _ := outer(dkAccum[pos], xNorm)
		dwvWeight, _, _ := outer(dvAccum[pos], xNorm)
		addGrad(g.gradMap["layer0.attn_wk"], dwkWeight)
		addGrad(g.gradMap["layer0.attn_wv"], dwvWeight)

		// Accumulate dx contributions at each past position from this attention
		for t := 0; t <= pos; t++ {
			dxKt := matVecMulTranspose(wk.data, dkPerPos[t], wk.rows, wk.cols)
			dxVt := matVecMulTranspose(wv.data, dvPerPos[t], wv.rows, wv.cols)
			for i := range dxFromAttn[t] {
				dxFromAttn[t][i] += dxKt[i] + dxVt[i]
			}
		}
		dxQ := matVecMulTranspose(wq.data, dqFull, wq.rows, wq.cols)
		for i := range dxFromAttn[pos] {
			dxFromAttn[pos][i] += dxQ[i]
		}

		// Residual path through rmsnorm
		dXAfterAttnNorm := gradRMSNorm(elemMul(dXAfterMlpNorm, g.stateDict["layer0.rms1_gamma"].data), cache.X[pos])
		addGradRMSNormGamma(dGamma1, dXAfterMlpNorm, cache.X[pos])
		dXResidual := make([]float64, nEmb)
		for i := range dXResidual {
			dXResidual[i] = dXAfterAttnNorm[i] + dxFromAttn[pos][i]
		}

		dxNorm := dXResidual
		dXResidualForGamma := make([]float64, nEmb)
		copy(dXResidualForGamma, dxNorm)
		dxNorm = gradRMSNorm(elemMul(dxNorm, g.stateDict["layer0.rms1_gamma"].data), cache.XRes[pos])
		addGradRMSNormGamma(dGamma1, dXResidualForGamma, cache.XRes[pos])

		tokenID := cache.Tokens[pos]
		wteGrad := g.gradMap["wte"]
		dtokEmb := make([]float64, nEmb)
		for i := range dtokEmb {
			dtokEmb[i] = dxNorm[i]
		}
		addEmbedGrad(wteGrad, tokenID, dtokEmb)

		wpeGrad := g.gradMap["wpe"]
		addEmbedGrad(wpeGrad, pos, dxNorm)
	}
	addGrad(g.gradMap["layer0.rms1_gamma"], dGamma1)
	addGrad(g.gradMap["layer0.rms2_gamma"], dGamma2)
}

func addGrad(m *matrix, grad []float64) {
	for i := range grad {
		m.data[i] += grad[i]
	}
}

func addEmbedGrad(m *matrix, row int, grad []float64) {
	start := row * m.cols
	for i := range grad {
		m.data[start+i] += grad[i]
	}
}
