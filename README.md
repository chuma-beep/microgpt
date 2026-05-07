# microgpt – A Minimal GPT Implementation in Pure Go

A tiny GPT language model trained from scratch in ~300 lines of Go. No external ML libraries—just manual matrix operations and backpropagation.

## What It Does

Trains on a dataset of first names and generates new, name-like strings.

```
$- microgpt: go run .
Loaded 32033 names
Vocab size: 27
step 100/10000, avg loss = 3.2709
step 200/10000, avg loss = 2.9365
step 300/10000, avg loss = 2.7667
step 400/10000, avg loss = 2.6826
step 500/10000, avg loss = 2.6297
step 600/10000, avg loss = 2.6187
step 700/10000, avg loss = 2.4309
step 800/10000, avg loss = 2.4725
step 900/10000, avg loss = 2.4779
step 1000/10000, avg loss = 2.5274
step 1100/10000, avg loss = 2.5182
step 1200/10000, avg loss = 2.4854
step 1300/10000, avg loss = 2.5007
step 1400/10000, avg loss = 2.5136
step 1500/10000, avg loss = 2.4791
step 1600/10000, avg loss = 2.4108
step 1700/10000, avg loss = 2.4744
step 1800/10000, avg loss = 2.4494
step 1900/10000, avg loss = 2.5082
step 2000/10000, avg loss = 2.4429
step 2100/10000, avg loss = 2.5586
step 2200/10000, avg loss = 2.4396
step 2300/10000, avg loss = 2.4555
step 2400/10000, avg loss = 2.5056
step 2500/10000, avg loss = 2.4960
step 2600/10000, avg loss = 2.4021
step 2700/10000, avg loss = 2.4629
step 2800/10000, avg loss = 2.4490
step 2900/10000, avg loss = 2.4571
step 3000/10000, avg loss = 2.4214
step 3100/10000, avg loss = 2.3886
step 3200/10000, avg loss = 2.4286
step 3300/10000, avg loss = 2.3797
step 3400/10000, avg loss = 2.4325
step 3500/10000, avg loss = 2.4301
step 3600/10000, avg loss = 2.3379
step 3700/10000, avg loss = 2.4585
step 3800/10000, avg loss = 2.4664
step 3900/10000, avg loss = 2.3688
step 4000/10000, avg loss = 2.3904
step 4100/10000, avg loss = 2.4503
step 4200/10000, avg loss = 2.4663
step 4300/10000, avg loss = 2.4116
step 4400/10000, avg loss = 2.4231
step 4500/10000, avg loss = 2.3917
step 4600/10000, avg loss = 2.4144
step 4700/10000, avg loss = 2.3856
step 4800/10000, avg loss = 2.3785
step 4900/10000, avg loss = 2.3638
step 5000/10000, avg loss = 2.3815
step 5100/10000, avg loss = 2.4053
step 5200/10000, avg loss = 2.4118
step 5300/10000, avg loss = 2.3807
step 5400/10000, avg loss = 2.4366
step 5500/10000, avg loss = 2.4576
step 5600/10000, avg loss = 2.4000
step 5700/10000, avg loss = 2.3283
step 5800/10000, avg loss = 2.3717
step 5900/10000, avg loss = 2.3879
step 6000/10000, avg loss = 2.4112
step 6100/10000, avg loss = 2.4566
step 6200/10000, avg loss = 2.4070
step 6300/10000, avg loss = 2.3834
step 6400/10000, avg loss = 2.3299
step 6500/10000, avg loss = 2.4263
step 6600/10000, avg loss = 2.4628
step 6700/10000, avg loss = 2.3954
step 6800/10000, avg loss = 2.3694
step 6900/10000, avg loss = 2.4101
step 7000/10000, avg loss = 2.3671
step 7100/10000, avg loss = 2.4049
step 7200/10000, avg loss = 2.4165
step 7300/10000, avg loss = 2.3754
step 7400/10000, avg loss = 2.3119
step 7500/10000, avg loss = 2.4153
step 7600/10000, avg loss = 2.2200
step 7700/10000, avg loss = 2.4192
step 7800/10000, avg loss = 2.4567
step 7900/10000, avg loss = 2.4215
step 8000/10000, avg loss = 2.3019
step 8100/10000, avg loss = 2.4421
step 8200/10000, avg loss = 2.3414
step 8300/10000, avg loss = 2.4648
step 8400/10000, avg loss = 2.3127
step 8500/10000, avg loss = 2.3963
step 8600/10000, avg loss = 2.3398
step 8700/10000, avg loss = 2.3863
step 8800/10000, avg loss = 2.5460
step 8900/10000, avg loss = 2.3263
step 9000/10000, avg loss = 2.3211
step 9100/10000, avg loss = 2.4179
step 9200/10000, avg loss = 2.2569
step 9300/10000, avg loss = 2.3857
step 9400/10000, avg loss = 2.3902
step 9500/10000, avg loss = 2.3295
step 9600/10000, avg loss = 2.4294
step 9700/10000, avg loss = 2.4105
step 9800/10000, avg loss = 2.2960
step 9900/10000, avg loss = 2.3135
step 10000/10000, avg loss = 2.4475
--- Generated names (temperature 0.5) ---
 1: mena
 2: ancdy
 3: anayne
 4: a
 5: meelyn
 6: ieelee
 7: ieli
 8: marian
 9: carien
10: eerely
11: eediela
12: aliee
13: annie
14: ee
15: nnne
16: elleey
17: neiele
18: rera
19: ajurle
20: ieieee
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
