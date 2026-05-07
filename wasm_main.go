//go:build js && wasm

package main

import (
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
