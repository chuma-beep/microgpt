# microgpt

A minimal GPT implementation written from scratch in Go. No external ML libraries — matrix operations, backpropagation, and the Adam optimizer are all implemented manually on `[]float64` slices.

Trained on 32,033 first names, the model learns to generate new name-like strings.

## Demo

```
$ go run . -steps 10000 -temperature 0.5
Loaded 32033 names
Vocab size: 27
step 100/10000,  avg loss = 3.2709
step 200/10000,  avg loss = 2.9365
step 300/10000,  avg loss = 2.7667
...
step 9900/10000, avg loss = 2.3135
step 10000/10000, avg loss = 2.4475

--- Generated names (temperature 0.5) ---
 1: mena
 2: marian
 3: carien
 4: annie
 5: anayne
 6: meelyn
 7: ajurle
 8: aliee
 9: rera
10: ancdy
```

Loss drops from ~3.27 (random chance) to ~2.20 over 10k steps. The model learns vowel patterns, name-like endings, and reasonable length — genuine pattern recognition with no hand-crafted rules.

## Usage

```bash
go run . -steps 10000 -temperature 0.5
```

On subsequent runs, saved weights are loaded automatically — no retraining needed:

```bash
go run .
# Loading weights from weights.bin
# --- Generated names (temperature 0.5) ---
```

Flags:
- `-steps` — training iterations (default: 10000)
- `-temperature` — sampling temperature (default: 0.5). Lower is more conservative, higher is more varied.
- `-weights` — path to weights file (default: weights.bin)

## Architecture

- **Vocabulary**: character-level, 26 letters + 1 BOS token = 27 tokens
- **Embeddings**: token (wte) + positional (wpe), 16 dimensions
- **Attention**: single transformer layer, 4-head causal self-attention
- **MLP**: fully connected, ReLU activation, 4× hidden dimension
- **Normalization**: RMSNorm
- **Optimizer**: Adam (β1=0.9, β2=0.999), lr=0.001
- **Parameters**: ~6k total

## Why no gonum

The point of this project is to understand what happens inside a transformer, not to call library functions. Every matrix multiply, softmax, RMSNorm, and backprop step is written by hand. This forces you to confront the actual math at each layer rather than treating it as a black box.

## Bugs found during implementation

### Wrong token in embedding backward

The backward pass was accumulating gradients into the embedding row of the *target* token instead of the *input* token:

```go
// Wrong
tokenID := cache.Targets[pos]  // target — what we're predicting

// Correct
tokenID := cache.Tokens[pos]   // input — what was actually embedded
```

The forward pass embeds the input token. The gradient must flow back to that same row. Using the target token sends the gradient to the wrong embedding entirely, so the embedding weights never learn correctly.

### Residual gradient corrupted by forward activation

In the attention block backward pass, the gradient accumulation for the residual connection was adding the forward activation value directly into the gradient:

```go
// Wrong
for i := range dXResidual {
    dXResidual[i] += xResidual[i]  // adds forward value, not gradient
}
```

Gradients and activations are completely different quantities. This injected the raw forward pass values into the gradient signal, producing incorrect updates throughout the network. Removing that line fixed it.

Both bugs produced the same symptom — loss bouncing at random-chance level (~3.3) with no downward trend — but for different reasons. The gradient debug print (`max grad`) was the diagnostic that revealed the backward pass was returning near-zero gradients.

## File structure

```
microgpt/
├── main.go          # CLI flags, calls Run()
├── train.go         # training loop, weight save/load, generation
├── model.go         # GPT struct, forward pass, backward pass
├── attention.go     # single-head attention forward
├── grad_attention.go # attention backward
├── cache.go         # activations stored for backward pass
├── tokenizer.go     # character-level tokenizer, BOS encoding
├── matutil.go       # matrix ops, softmax, rmsnorm, relu
└── adam.go          # Adam optimizer
```

---

