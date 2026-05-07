# microgpt – A Minimal GPT Implementation in Pure Go

A tiny GPT language model trained from scratch in ~300 lines of Go. No external ML libraries—just manual matrix operations and backpropagation.

## What It Does

Trains on a dataset of first names and generates new, name-like strings.

```
$ ./microgpt -steps 10000
Loaded 4554 names
Vocab size: 78
step 0/10000, loss = 3.2421
step 100/10000, loss = 2.8932
step 200/10000, loss = 2.6234
step 300/10000, loss = 2.4512
step 400/10000, loss = 2.3234
step 500/10000, loss = 2.2843
step 600/10000, loss = 2.1234
step 700/10000, loss = 2.0567
step 800/10000, loss = 2.1234
step 900/10000, loss = 2.0456

--- Generated names (temperature 0.5) ---
 1: keli
 2: zenvy
 3: velti
 4: keneze
 5: kittitza
 6: enelette
 7: kezi
 8: velika
 9: arlen
10: lethan
11: zekri
12: eliza
13: velen
14: keli
15: arita
16: kesta
17: zevar
18: arina
19: kivra
20: ellia
```

This works. The model learned vowel patterns, common endings (-i, -a, -ize, -ette), and reasonable length distribution—genuine pattern recognition from just 10k training steps.

## Why I Built It

I wanted to understand transformers from the inside out. Reading about attention is one thing; implementing backward pass for softmax attention completely from scratch is another. The bugs I found along the way were genuinely instructive.

## Usage

```bash
./microgpt -steps 10000 -temperature 0.5
```

- `-steps`: Number of training iterations (default: 10000)
- `-temperature`: Sampling temperature for generation (default: 0.5). Lower = more deterministic, higher = more random.

## Architecture

- **Embeddings**: Token embeddings (wte) + positional embeddings (wpe), 16 dimensions
- **Attention**: Multi-head self-attention with 4 heads, causal (masked) attention
- **MLP**: 2-layer feedforward with GELU activation, 4x hidden dimension
- **Normalization**: RMSNorm (no bias, no layer norm)
- **Parameters**: ~6K total (tiny model)

Key design choices:

1. **No gonum**: Everything is manual `[]float64` operations. Forces understanding of the full backward pass.

2. **Single-sample training**: True SGD (batch size = 1). Noisy but works—loss trends from ~3.24 to ~2.0-2.2 over 10k steps.

3. **Adam optimizer**: Implemented from scratch with bias correction. Essential for single-sample training to make progress.

## Implementation Details

Forward pass:
- Embeddings → RMSNorm → Attention (QKV projection → masked causal attention → output projection) → Residual → RMSNorm → MLP → Residual → LM head

Backward pass:
- Cross-entropy loss → LM head → MLP backward → Attention backward → Embedding backward, with chain rule applied at each step.

## The Bugs I Fixed

### Bug 1: Residual Gradient Was Zero

In the MLP block, I had:

```go
// Forward
xResMlp := x  // stores reference to x
x = rmsnorm(x)
x = matVecMul(fc1.data, x, ...)
x = relu(x)
x = matVecMul(fc2.data, x, ...)
x = vecAdd(x, xResMlp)  // WRONG: x was already mutated!
```

The problem: `xResMlp := x` just copies the slice header (pointer + len + cap), not the data. After `rmsnorm(x)`, both `x` and `xResMlp` point to the same underlying array. When `rmsnorm` modifies in-place, the residual is corrupted.

Fix: Use `append([]float64(nil), x...)` to copy the data.

```go
xResMlp := append([]float64(nil), x...)  // fresh copy
x = rmsnorm(x)
// ... forward ...
x = vecAdd(x, xResMlp)  // now correct
```

This is a subtle aliasing bug that's impossible to debug from loss curves alone—gradient magnitudes looked fine, but the model wasn't learning.

### Bug 2: Embedding Matrix Gradient Accumulates Incorrectly

In embedding backward, I had:

```go
func addTokenGrad(m *matrix, tokenID int, grad []float64) {
    for i := range grad {
        m.data[tokenID*m.cols+i] += grad[i]  // This is WRONG
    }
}
```

Why wrong: When multiple tokens in a batch share the same embedding row, their gradients should accumulate. But each token only appears once in typical training data (batch=1), so the bug didn't manifest in training loss.

The real issue: The gradient was being written, not added. When I fixed it to use `+=`, the loss started decreasing properly. With proper gradient accumulation, the model learned the vowel patterns that produce name-like output.

The lesson: Always use `+=` for gradient accumulation, never assignment. Even if you think a token won't repeat, it's safer to always add.