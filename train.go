// train.go – training loop, inference, and generation
package main

import (
	"encoding/binary"
	"fmt"
	"math"
	"math/rand/v2"
	"os"
)

func saveWeights(path string, params []*matrix) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, m := range params {
		if err := binary.Write(f, binary.LittleEndian, m.data); err != nil {
			return err
		}
	}
	return nil
}

func loadWeights(path string, params []*matrix) error {
	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()
	for _, m := range params {
		if err := binary.Read(f, binary.LittleEndian, m.data); err != nil {
			return err
		}
	}
	return nil
}

func Run(steps int, genTemp float64, weightsPath string) {
	// Load data
	names, err := LoadNames("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
	if err != nil {
		panic(err)
	}
	fmt.Printf("Loaded %d names\n", len(names))

	// Tokenizer
	tok := NewTokenizer(names)
	fmt.Printf("Vocab size: %d\n", tok.VocabSize)

	// Model
	gpt := NewGPT(tok)

	// Load weights if file exists
	if _, err := os.Stat(weightsPath); err == nil {
		fmt.Printf("Loading weights from %s\n", weightsPath)
		if err := loadWeights(weightsPath, gpt.params); err != nil {
			panic(err)
		}
	} else {
		// Flatten parameters
		paramsFlat := make([]float64, 0)
		for _, m := range gpt.params {
			paramsFlat = append(paramsFlat, m.data...)
		}
		gradsFlat := make([]float64, len(paramsFlat))

		// Adam
		adam := NewAdam(len(paramsFlat))

		baseLR := 0.001

		var accumLoss float64
		var accumCount int

		for step := 0; step < steps; step++ {
			doc := names[step%len(names)]
			tokens := tok.Encode(doc)
			n := min(blockSize, len(tokens)-1)
			if n < 1 {
				continue
			}
			loss, cache := gpt.ForwardSeq(tokens[:n+1])

			// Backward
			gpt.Backward(cache)

			// Flatten gradients
			idx := 0
			for _, gr := range gpt.grads {
				copy(gradsFlat[idx:], gr.data)
				idx += len(gr.data)
			}

			lr := baseLR
			adam.Update(paramsFlat, gradsFlat, lr)

			// Copy back
			idx = 0
			for _, m := range gpt.params {
				copy(m.data, paramsFlat[idx:idx+len(m.data)])
				idx += len(m.data)
			}

			// Zero grads
			for _, gr := range gpt.grads {
				for i := range gr.data {
					gr.data[i] = 0.0
				}
			}

			accumLoss += loss
			accumCount++

			if step%100 == 99 {
				avg := accumLoss / float64(accumCount)
				fmt.Printf("step %d/%d, avg loss = %.4f\n", step+1, steps, avg)
				accumLoss = 0
				accumCount = 0
			}
		}

		// Save weights
		fmt.Printf("Saving weights to %s\n", weightsPath)
		if err := saveWeights(weightsPath, gpt.params); err != nil {
			panic(err)
		}
	}

	// Inference
	fmt.Printf("\n--- Generated names (temperature %.1f) ---\n", genTemp)
	for i := 0; i < 20; i++ {
		name := gpt.Generate(genTemp)
		fmt.Printf("%2d: %s\n", i+1, name)
	}
}

// Generate samples using the trained GPT
func (g *GPT) Generate(temperature float64) string {
	tok := g.tok
	token := tok.BOS
	sample := make([]byte, 0)

	// We'll reuse forward step logic without caching
	// Simplified: we only need to run one token at a time, but we must maintain KV caches.
	keysCache := make([][][]float64, nLayer)
	valuesCache := make([][][]float64, nLayer)

	for pos := 0; pos < blockSize; pos++ {
		// Get token embedding and position embedding
		wte := g.stateDict["wte"]
		wpe := g.stateDict["wpe"]
		tokEmb := wte.row(token)
		posEmb := wpe.row(pos)
		x := vecAdd(tokEmb, posEmb)
		x = rmsnorm(x)

		// Attention block
		xResAttn := append([]float64(nil), x...)
		x = rmsnorm(x)

		wq := g.stateDict["layer0.attn_wq"]
		wk := g.stateDict["layer0.attn_wk"]
		wv := g.stateDict["layer0.attn_wv"]
		q := matVecMul(wq.data, x, wq.rows, wq.cols)
		k := matVecMul(wk.data, x, wk.rows, wk.cols)
		v := matVecMul(wv.data, x, wv.rows, wv.cols)

		keysCache[0] = append(keysCache[0], k)
		valuesCache[0] = append(valuesCache[0], v)

		attnOut := make([]float64, nEmb)
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
			headOut, _ := attentionHead(qHead, kHeads, vHeads, headDim)
			for j := 0; j < headDim; j++ {
				attnOut[hs+j] = headOut[j]
			}
		}
		wo := g.stateDict["layer0.attn_wo"]
		attnProj := matVecMul(wo.data, attnOut, wo.rows, wo.cols)
		x = vecAdd(attnProj, xResAttn)

		// MLP block
		xResMlp := append([]float64(nil), x...)
		x = rmsnorm(x)
		fc1 := g.stateDict["layer0.mlp_fc1"]
		fc2 := g.stateDict["layer0.mlp_fc2"]
		x = matVecMul(fc1.data, x, fc1.rows, fc1.cols)
		x = relu(x)
		x = matVecMul(fc2.data, x, fc2.rows, fc2.cols)
		x = vecAdd(x, xResMlp)

		lmHead := g.stateDict["lm_head"]
		logits := matVecMul(lmHead.data, x, lmHead.rows, lmHead.cols)

		// Temperature sampling
		probs := softmax(logits)
		if temperature != 1.0 {
			for i := range probs {
				probs[i] = math.Pow(probs[i], 1.0/temperature)
			}
			sum := 0.0
			for _, p := range probs {
				sum += p
			}
			for i := range probs {
				probs[i] /= sum
			}
		}
		next := sampleMultinomial(probs)
		if next == tok.BOS {
			break
		}
		sample = append(sample, tok.IdxToChar[next])
		token = next
	}
	return string(sample)
}

func sampleMultinomial(probs []float64) int {
	r := rand.Float64()
	cum := 0.0
	for i, p := range probs {
		cum += p
		if r < cum {
			return i
		}
	}
	return len(probs) - 1
}
