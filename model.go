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
		x = rmsnorm(x)
		cache.X[pos] = append([]float64(nil), x...)

		// Attention block
		xResAttn := append([]float64(nil), x...)
		x = rmsnorm(x)

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
		x = rmsnorm(x)
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
		dlmHeadWeight := outerProdVecMat(dlogits, cache.FinalX[pos])
		addGrad(g.gradMap["lm_head"], dlmHeadWeight)
		dxAfterResidual := matVecMulT(lmHead.data, dlogits, lmHead.rows, lmHead.cols)

		dx := dxAfterResidual

		fc2 := g.stateDict["layer0.mlp_fc2"]
		dfc2Weight := outerProdVecMat(dx, cache.MLPReLU[pos])
		dfc2Input := matVecMulT(fc2.data, dx, fc2.rows, fc2.cols)
		addGrad(g.gradMap["layer0.mlp_fc2"], dfc2Weight)

		drelu := make([]float64, len(dfc2Input))
		for i, v := range dfc2Input {
			if cache.MLPReLU[pos][i] > 0 {
				drelu[i] = v
			}
		}

		fc1 := g.stateDict["layer0.mlp_fc1"]
		dfc1Weight := outerProdVecMat(drelu, cache.MLPIn[pos])
		dfc1Input := matVecMulT(fc1.data, drelu, fc1.rows, fc1.cols)
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
		dXAfterMlpNorm := rmsnormBackward(dXResMlp, xBeforeMlpNorm)

		attnOut := cache.AttnConcat[pos]
		wo := g.stateDict["layer0.attn_wo"]
		_ = matVecMulT(wo.data, dXAfterMlpNorm, wo.rows, wo.cols)
		dwoWeight := outerProdVecMat(dXAfterMlpNorm, attnOut)
		addGrad(g.gradMap["layer0.attn_wo"], dwoWeight)

		dXAfterAttn := make([]float64, nEmb)
		for i := range dXAfterAttn {
			dXAfterAttn[i] = dXAfterMlpNorm[i]
		}

		dXAfterAttnNorm := make([]float64, nEmb)
		for i := range dXAfterAttnNorm {
			dXAfterAttnNorm[i] = dXAfterAttn[i]
		}

		xAfterAttnNorm := cache.X[pos]
		dXAfterAttnNorm = rmsnormBackward(dXAfterAttnNorm, xAfterAttnNorm)

		dXResidual := make([]float64, nEmb)
		for i := range dXResidual {
			dXResidual[i] = dXAfterAttnNorm[i]
		}

		wq := g.stateDict["layer0.attn_wq"]
		wk := g.stateDict["layer0.attn_wk"]
		wv := g.stateDict["layer0.attn_wv"]

		dq := matVecMulT(wq.data, dXAfterAttnNorm, wq.rows, wq.cols)
		dk := matVecMulT(wk.data, dXAfterAttnNorm, wk.rows, wk.cols)
		dv := matVecMulT(wv.data, dXAfterAttnNorm, wv.rows, wv.cols)

		xNorm := cache.X[pos]
		dwqWeight := outerProdVecMat(dq, xNorm)
		dwkWeight := outerProdVecMat(dk, xNorm)
		dwvWeight := outerProdVecMat(dv, xNorm)

		addGrad(g.gradMap["layer0.attn_wq"], dwqWeight)
		addGrad(g.gradMap["layer0.attn_wk"], dwkWeight)
		addGrad(g.gradMap["layer0.attn_wv"], dwvWeight)

		dxNorm := dXResidual
		dxNorm = rmsnormBackward(dxNorm, cache.XRes[pos])

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
}

func addGrad(m *matrix, grad []float64) {
	for i := range grad {
		m.data[i] += grad[i]
	}
}

func outerProdVecMat(a, b []float64) []float64 {
	result := make([]float64, len(a)*len(b))
	rows := len(a)
	cols := len(b)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[i*cols+j] = a[i] * b[j]
		}
	}
	return result
}

func matVecMulT(mat []float64, vec []float64, rows, cols int) []float64 {
	result := make([]float64, cols)
	for i := 0; i < rows; i++ {
		for j := 0; j < cols; j++ {
			result[j] += mat[i*cols+j] * vec[i]
		}
	}
	return result
}

func rmsnormBackward(dy, x []float64) []float64 {
	dx := make([]float64, len(x))
	mean := 0.0
	for _, v := range x {
		mean += v * v
	}
	mean /= float64(len(x))
	mean = math.Sqrt(mean + 1e-8)
	invMean := 1.0 / mean

	for i := range dx {
		dx[i] = invMean * dy[i]
		dx[i] -= invMean * dy[i] * x[i] * x[i] / (float64(len(x)) * mean * mean)
	}

	sum := 0.0
	for i := range dy {
		sum += x[i] * dy[i] * invMean
	}
	for i := range dx {
		dx[i] -= x[i] * sum / (float64(len(x)) * mean * mean)
	}

	return dx
}

func addEmbedGrad(m *matrix, row int, grad []float64) {
	start := row * m.cols
	for i := range grad {
		m.data[start+i] += grad[i]
	}
}
