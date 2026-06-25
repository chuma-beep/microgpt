//go:build js && wasm

package main

import (
	"encoding/json"
	"sort"
	"syscall/js"
)

var globalTrainer *Trainer
var globalTok *Tokenizer
var globalGPT *GPT

func main() {
	c := make(chan struct{}, 0)
	js.Global().Set("goInit", js.FuncOf(goInit))
	js.Global().Set("goTrainStep", js.FuncOf(goTrainStep))
	js.Global().Set("goGenerate", js.FuncOf(goGenerate))
	js.Global().Set("goGenerateWithProbs", js.FuncOf(goGenerateWithProbs))
	js.Global().Set("goGetWTE", js.FuncOf(goGetWTE))
	js.Global().Set("goGetWPE", js.FuncOf(goGetWPE))
	js.Global().Set("goAttentionWeights", js.FuncOf(goAttentionWeights))
	js.Global().Set("goStepThrough", js.FuncOf(goStepThrough))
	js.Global().Set("goLogitLens", js.FuncOf(goLogitLens))
	js.Global().Set("goGetWeightMatrix", js.FuncOf(goGetWeightMatrix))
	js.Global().Set("goInitCustom", js.FuncOf(goInitCustom))
	<-c
}

func goInit(this js.Value, args []js.Value) interface{} {
	if len(args) == 0 {
		return "error: callback required"
	}
	callback := args[0]
	go func() {
		names, err := LoadNames("https://raw.githubusercontent.com/karpathy/makemore/master/names.txt")
		if err != nil {
			callback.Invoke("error: "+err.Error(), js.Null())
			return
		}
		splitIdx := int(float64(len(names)) * 0.9)
		trainNames := names[:splitIdx]
		globalTok = NewTokenizer(names)
		globalGPT = NewGPT(globalTok)
		globalTrainer = NewTrainer(globalGPT, globalTok, trainNames)
		callback.Invoke(js.Null(), "ok")
	}()
	return nil
}

func goTrainStep(this js.Value, args []js.Value) interface{} {
	if globalTrainer == nil {
		return js.ValueOf(-1.0)
	}
	loss := globalTrainer.Step()
	return js.ValueOf(loss)
}

func goGenerate(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil {
		return js.ValueOf("")
	}
	temperature := 0.5
	if len(args) > 0 {
		temperature = args[0].Float()
	}
	return js.ValueOf(globalGPT.Generate(temperature))
}

func goGenerateWithProbs(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil {
		return js.Undefined()
	}
	temperature := 0.5
	if len(args) > 0 {
		temperature = args[0].Float()
	}
	name, probs := globalGPT.GenerateWithProbs(temperature)
	result := make(map[string]interface{})
	result["name"] = name
	probsFloat := make([]interface{}, len(probs))
	for i, p := range probs {
		probsFloat[i] = p
	}
	result["probs"] = probsFloat
	return js.ValueOf(result)
}

func goGetWTE(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil {
		return js.Undefined()
	}
	wte := globalGPT.stateDict["wte"]
	result := make([]interface{}, len(wte.data))
	for i, v := range wte.data {
		result[i] = v
	}
	return js.ValueOf(result)
}

func goGetWPE(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil {
		return js.Undefined()
	}
	wpe := globalGPT.stateDict["wpe"]
	result := make([]interface{}, len(wpe.data))
	for i, v := range wpe.data {
		result[i] = v
	}
	return js.ValueOf(result)
}

func goAttentionWeights(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil || len(args) == 0 {
		return js.Undefined()
	}
	name := args[0].String()
	tokens := globalGPT.tok.Encode(name)
	n := min(blockSize, len(tokens)-1)
	if n < 1 {
		return js.Undefined()
	}
	_, cache := globalGPT.ForwardSeq(tokens[:n+1])

	flat := make([]interface{}, nHead*n*n)
	for pos := 0; pos < n; pos++ {
		weights := cache.AttnWeights[pos]
		for h := 0; h < nHead; h++ {
			for j := 0; j <= pos; j++ {
				idx := h*n*n + pos*n + j
				flat[idx] = weights[h*(pos+1)+j]
			}
		}
	}
	return js.ValueOf(flat)
}

type stepThroughStage struct {
	Name string    `json:"name"`
	Dims []int     `json:"dims"`
	Data []float64 `json:"data"`
}

type stepThroughData struct {
	NPositions int                `json:"nPositions"`
	NEmb       int                `json:"nEmb"`
	HeadDim    int                `json:"headDim"`
	NHead      int                `json:"nHead"`
	NVocab     int                `json:"nVocab"`
	BlockSize  int                `json:"blockSize"`
	Tokens     []int              `json:"tokens"`
	Targets    []int              `json:"targets"`
	Labels     []string           `json:"labels"`
	Stages     []stepThroughStage `json:"stages"`
}

func goStepThrough(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil || len(args) == 0 {
		return `{"error":"model not initialized"}`
	}
	name := args[0].String()
	tokens := globalGPT.tok.Encode(name)
	n := min(blockSize, len(tokens)-1)
	if n < 1 {
		return `{"error":"sequence too short"}`
	}
	_, cache := globalGPT.ForwardSeq(tokens[:n+1])

	labels := make([]string, n)
	for i := 0; i < n; i++ {
		tok := cache.Tokens[i]
		if tok == globalGPT.tok.BOS {
			labels[i] = "BOS"
		} else {
			labels[i] = string([]byte{globalGPT.tok.IdxToChar[tok]})
		}
	}

	stages := []stepThroughStage{
		{
			Name: "Token + Position Embedding",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.XRes, nEmb, 1),
		},
		{
			Name: "RMSNorm",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.X, nEmb, 1),
		},
		{
			Name: "QKV Projections",
			Dims: []int{nEmb, nEmb, nEmb},
			Data: flattenQKV(cache.Q, cache.K, cache.V),
		},
		{
			Name: "Attention Scores",
			Dims: []int{nHead, posLenArr(cache.AttnScores, n)},
			Data: flattenVecs(cache.AttnScores, 0, 0),
		},
		{
			Name: "Attention Weights",
			Dims: []int{nHead, posLenArr(cache.AttnWeights, n)},
			Data: flattenVecs(cache.AttnWeights, 0, 0),
		},
		{
			Name: "Attention Output",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.AttnConcat, nEmb, 1),
		},
		{
			Name: "WO Projection",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.AttnProj, nEmb, 1),
		},
		{
			Name: "RMSNorm (MLP)",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.MLPIn, nEmb, 1),
		},
		{
			Name: "MLP fc1 (64d)",
			Dims: []int{4 * nEmb},
			Data: flattenVecs(cache.MLPPreReLU, 4*nEmb, 1),
		},
		{
			Name: "MLP ReLU (64d)",
			Dims: []int{4 * nEmb},
			Data: flattenVecs(cache.MLPReLU, 4*nEmb, 1),
		},
		{
			Name: "MLP fc2 + Residual",
			Dims: []int{nEmb},
			Data: flattenVecs(cache.FinalX, nEmb, 1),
		},
		{
			Name: "Logits",
			Dims: []int{globalGPT.tok.VocabSize},
			Data: flattenVecs(cache.Logits, globalGPT.tok.VocabSize, 1),
		},
		{
			Name: "Softmax / Probabilities",
			Dims: []int{globalGPT.tok.VocabSize},
			Data: flattenVecs(cache.Probs, globalGPT.tok.VocabSize, 1),
		},
	}

	data := stepThroughData{
		NPositions: n,
		NEmb:       nEmb,
		HeadDim:    headDim,
		NHead:      nHead,
		NVocab:     globalGPT.tok.VocabSize,
		BlockSize:  blockSize,
		Tokens:     cache.Tokens,
		Targets:    cache.Targets,
		Labels:     labels,
		Stages:     stages,
	}
	b, err := json.Marshal(data)
	if err != nil {
		return `{"error":"` + err.Error() + `"}`
	}
	return string(b)
}

func flattenVecs(rows [][]float64, dimsPerRow int, numSubVectors int) []float64 {
	n := len(rows)
	stride := dimsPerRow / max(1, numSubVectors)
	total := n * stride * max(1, numSubVectors)
	out := make([]float64, 0, total)
	for i := 0; i < n; i++ {
		if rows[i] != nil {
			out = append(out, rows[i]...)
		}
	}
	return out
}

func flattenQKV(qs, ks, vs [][]float64) []float64 {
	n := len(qs)
	out := make([]float64, 0, n*nEmb*3)
	for i := 0; i < n; i++ {
		if qs[i] != nil {
			out = append(out, qs[i]...)
		}
		if ks[i] != nil {
			out = append(out, ks[i]...)
		}
		if vs[i] != nil {
			out = append(out, vs[i]...)
		}
	}
	return out
}

func posLenArr(slices [][]float64, nPos int) int {
	for i := 0; i < nPos; i++ {
		if slices[i] != nil && len(slices[i]) > 0 {
			return nPos // each position has nHead*(pos+1) entries; return nPos as dims hint for JS
		}
	}
	return nPos
}

type tokenProb struct {
	Idx  int     `json:"idx"`
	Char string  `json:"char"`
	Prob float64 `json:"prob"`
}

type logitLensData struct {
	NPositions int          `json:"nPositions"`
	Tokens     []int        `json:"tokens"`
	Targets    []int        `json:"targets"`
	Labels     []string     `json:"labels"`
	TopK       [][]tokenProb `json:"topK"`
}

func goLogitLens(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil || len(args) == 0 {
		return `{"error":"model not initialized"}`
	}
	prefix := args[0].String()
	tokens := globalGPT.tok.Encode(prefix)
	n := min(blockSize, len(tokens)-1)
	if n < 1 {
		return `{"error":"sequence too short"}`
	}
	_, cache := globalGPT.ForwardSeq(tokens[:n+1])

	labels := make([]string, n)
	for i := 0; i < n; i++ {
		tok := cache.Tokens[i]
		if tok == globalGPT.tok.BOS {
			labels[i] = "BOS"
		} else {
			labels[i] = string([]byte{globalGPT.tok.IdxToChar[tok]})
		}
	}

	topK := make([][]tokenProb, n)
	for pos := 0; pos < n; pos++ {
		probs := cache.Probs[pos]
		indexed := make([]tokenProb, len(probs))
		for j := range probs {
			char := ""
			if j == globalGPT.tok.BOS {
				char = "BOS"
			} else {
				char = string([]byte{globalGPT.tok.IdxToChar[j]})
			}
			indexed[j] = tokenProb{Idx: j, Char: char, Prob: probs[j]}
		}
		sort.Slice(indexed, func(a, b int) bool {
			return indexed[a].Prob > indexed[b].Prob
		})
		k := min(5, len(indexed))
		topK[pos] = indexed[:k]
	}

	data := logitLensData{
		NPositions: n,
		Tokens:     cache.Tokens,
		Targets:    cache.Targets,
		Labels:     labels,
		TopK:       topK,
	}
	b, err := json.Marshal(data)
	if err != nil {
		return `{"error":"` + err.Error() + `"}`
	}
	return string(b)
}

func goGetWeightMatrix(this js.Value, args []js.Value) interface{} {
	if globalGPT == nil || len(args) == 0 {
		return js.Undefined()
	}
	paramName := args[0].String()
	m, ok := globalGPT.stateDict[paramName]
	if !ok {
		return js.Undefined()
	}
	result := make([]interface{}, len(m.data))
	for i, v := range m.data {
		result[i] = v
	}
	return js.ValueOf(result)
}

func goInitCustom(this js.Value, args []js.Value) interface{} {
	if len(args) == 0 {
		return "error: data required"
	}
	dataJSON := args[0].String()
	var data []string
	if err := json.Unmarshal([]byte(dataJSON), &data); err != nil {
		return "error: invalid JSON: " + err.Error()
	}
	if len(data) < 2 {
		return "error: need at least 2 names"
	}
	splitIdx := int(float64(len(data)) * 0.9)
	if splitIdx < 1 {
		splitIdx = 1
	}
	trainNames := data[:splitIdx]
	globalTok = NewTokenizer(data)
	globalGPT = NewGPT(globalTok)
	globalTrainer = NewTrainer(globalGPT, globalTok, trainNames)
	return "ok"
}
