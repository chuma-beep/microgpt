<div align="center">



                            
    
       ███╗   ███╗██╗ ██████╗██████╗  ██████╗  ██████╗ ██████╗ ████████╗
       ████╗ ████║██║██╔════╝██╔══██╗██╔═══██╗██╔════╝ ██╔══██╗╚══██╔══╝
       ██╔████╔██║██║██║     ██████╔╝██║   ██║██║  ███╗██████╔╝   ██║   
       ██║╚██╔╝██║██║██║     ██╔══██╗██║   ██║██║   ██║██╔═══╝    ██║   
       ██║ ╚═╝ ██║██║╚██████╗██║  ██║╚██████╔╝╚██████╔╝██║        ██║   
       ╚═╝     ╚═╝╚═╝ ╚═════╝╚═╝  ╚═╝ ╚═════╝  ╚═════╝ ╚═╝        ╚═╝   
  

**A minimal GPT implementation written from scratch in Go.**  
No external ML libraries. Every matrix multiply, backprop step, and optimizer update is done by hand.



![Go](https://img.shields.io/badge/Go-1.21+-00ADD8?style=flat-square&logo=go&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-6e7681?style=flat-square)
![Parameters](https://img.shields.io/badge/Parameters-~4.2k-f0883e?style=flat-square)
![Vocab](https://img.shields.io/badge/Vocab-27_tokens-3fb950?style=flat-square)

<br>

</div>

---

Trained on 32,033 first names, the model learns to generate new name-like strings using a single-layer transformer with character-level tokenization. The goal is transparency over performance: no `gonum`, no tensor libraries, just raw `[]float64` arithmetic.

---

## Demo

```console
$ go run . -steps 10000 -temperature 0.5

Loaded 32033 names
Vocab size: 27
Train: 28829, Val: 3204

step  100/10000   avg loss = 3.0746
step  200/10000   avg loss = 2.5897
step  300/10000   avg loss = 2.3875
step  500/10000   avg loss = 2.3157   val loss = 2.7141
step 1000/10000   avg loss = 2.3025   val loss = 2.6430

--- Generated names (temperature 0.5) ---
  1: derinne       11: mare
  2: ennna         12: kylie
  3: elynna        13: rita
  4: daylee        14: lilia
  5: erita         15: nora
  6: gelen         16: sienna
  7: elos          17: ari
  8: alynn         18: leona
  9: anna          19: alina
 10: danaya        20: miri
```

Train loss drops from ~3.27 (random chance) to ~2.30 over 1000 steps. The gap between train and val (~0.3) indicates mild overfitting; the model generalizes but could benefit from more data or regularization.

---

## Usage

```bash
go run . -steps 10000 -temperature 0.5
```

On subsequent runs, saved weights are loaded automatically. Pass `-generate` to skip retraining:

```bash
go run . -generate
# Loading weights from weights.bin
# --- Generated names (temperature 0.5) ---
```

**Flags**

| Flag | Default | Description |
|------|---------|-------------|
| `-steps` | `10000` | Training iterations |
| `-temperature` | `0.5` | Sampling temperature. Lower is conservative, higher is varied |
| `-weights` | `weights.bin` | Path to weights file |
| `-generate` | | Skip training; generate from saved weights only |

---

## Architecture

| Component | Detail |
|-----------|--------|
| **Vocabulary** | Character-level: 26 letters + 1 BOS token = 27 tokens |
| **Embeddings** | Token (wte) + positional (wpe), 16 dimensions |
| **Attention** | Single transformer layer, 4-head causal self-attention |
| **MLP** | Fully connected, ReLU activation, 4x hidden dimension |
| **Normalization** | RMSNorm |
| **Optimizer** | Adam (b1=0.9, b2=0.999, lr=0.001) |
| **Parameters** | ~4.2k total (4,224 with default vocab, incl. 2x16 RMSNorm gamma) |

---

## WASM / Browser

A WebAssembly build target is provided via `wasm_main.go` (`//go:build js && wasm`):

```bash
GOOS=js GOARCH=wasm go build -o microgpt.wasm
```

The `viz/` directory contains a React + Vite + TypeScript frontend for interactive browser-based training and generation.

---

## Why no gonum

The point of this project is to understand what happens inside a transformer, not to call library functions. Every matrix multiply, softmax, RMSNorm, and backprop step is written by hand. This forces you to confront the actual math at each layer rather than treating it as a black box.

---

## Bugs found during implementation

### Wrong token in embedding backward

The backward pass was accumulating gradients into the embedding row of the *target* token instead of the *input* token:

```go
// Wrong: targets what we're predicting, not what was embedded
tokenID := cache.Targets[pos]

// Correct: inputs are what was actually embedded
tokenID := cache.Tokens[pos]
```

The forward pass embeds the input token, so the gradient must flow back to that same row. Using the target token sends the gradient to the wrong embedding entirely; the embedding weights never learn correctly.

---

### Residual gradient corrupted by forward activation

In the attention block backward pass, the gradient accumulation for the residual connection was adding the forward activation value directly into the gradient:

```go
// Wrong: adds the raw forward value, not a gradient
for i := range dXResidual {
    dXResidual[i] += xResidual[i]
}
```

Gradients and activations are completely different quantities. This injected raw forward pass values into the gradient signal, producing incorrect updates throughout the network. Removing that line fixed it.

Both bugs produced the same symptom: loss bouncing at random-chance level (~3.3) with no downward trend. The diagnostic was the `max grad` debug print, which revealed the backward pass was returning near-zero gradients.

---

## File structure

```
microgpt/
├── main.go             CLI flags, calls Run()
├── data.go             Download and parse names.txt
├── train.go            Training loop, weight save/load, generation
├── model.go            GPT struct, forward pass, backward pass
├── attention.go        Single-head attention forward
├── grad_attention.go   Attention backward
├── cache.go            Activations stored for backward pass
├── tokenizer.go        Character-level tokenizer, BOS encoding
├── matutil.go          Matrix ops, softmax, rmsnorm, relu
├── adam.go             Adam optimizer
├── wasm_main.go        WebAssembly build target (go:build js)
├── viz/                React/Vite visualization frontend
├── *_test.go           Unit and gradient-check tests
├── go.mod
├── go.sum
├── .gitignore
└── weights.bin
```

---

## Testing

```bash
go test ./... -v
```

**Coverage**

| Area | What is tested |
|------|---------------|
| **Math primitives** | Softmax, RMSNorm, ReLU, matrix-vector ops, outer product against hand-computed values |
| **Gradient checks** | Finite-difference validation for RMSNorm, linear layer, and single-head attention backward |
| **Model** | Directional gradient check on full `ForwardSeq` + `Backward`, param dimension verification, all-gradients-nonzero |
| **Optimizer** | Adam update correctness, step direction, momentum/velocity tracking |
| **Tokenizer** | Encode/decode roundtrip, BOS wrapping, edge cases (empty input, single char) |
| **Weights** | Save/load roundtrip binary fidelity |
| **Convergence** | End-to-end training on small dataset; loss must decrease |

---

<div align="center">
<sub>Built without ML libraries. All math by hand.</sub>
</div>
