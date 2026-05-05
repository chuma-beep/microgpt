# AGENT.md — microgpt-go

This file defines the rules, decisions, and conventions for this project.
All AI agents and contributors must follow these exactly.

---

## Project

A from-scratch Go implementation of Andrej Karpathy's microGPT (makemore variant).
Character-level language model trained on names.txt.
Goal: understand transformer internals + produce a strong portfolio project.

---

## Structure

Flat package layout. All files in root, all under `package main`.
Do NOT introduce sub-packages unless explicitly instructed.

```
microgpt-go/
├── main.go
├── data.go
├── tokenizer.go
├── matutil.go
├── model.go
├── attention.go
├── grad_attention.go
├── cache.go
├── adam.go
└── train.go
```

---

## Hard Rules

- NO external dependencies except the Go standard library
- NO gonum, NO blas, NO tensor libraries
- All matrix operations are implemented manually on `[]float64`
- Do not use `math/rand` (v1) — use `math/rand/v2` only
- Do not call `rand.Seed` — v2 seeds automatically
- All files use `package main`
- Do not add `func main()` to any file except `main.go`

---

## Conventions

### Types
- Matrix: `[]float64` with explicit `rows` and `cols` passed alongside
- No matrix struct wrapper unless explicitly introduced
- Byte-level tokenization using `byte`, not `rune` (names.txt is ASCII)

### Naming
- Exported functions use PascalCase
- Internal helpers use camelCase
- Struct fields use PascalCase

### Error handling
- Return `error` from functions that can fail (IO, HTTP)
- Use `panic` only for unrecoverable programmer errors (bad dimensions, etc.)
- No silent failures

### Comments
- Each file has a one-line comment at the top describing its responsibility
- Non-obvious math steps must have a comment explaining what is being computed

---

## Tokenizer Rules

- `CharToIdx map[byte]int` — maps character byte to index
- `IdxToChar []byte` — reverse lookup slice
- `BOS int` — index of the special begin/end-of-sequence token
- `VocabSize int` — total vocab including BOS (= number of unique chars + 1)
- Unique bytes MUST be sorted before index assignment (for reproducibility)
- `Encode(s string) []int` returns `[BOS, ...char indices..., BOS]`
- `Decode(ids []int) string` skips BOS tokens, returns the name string
- Expected VocabSize for names.txt: 27 (26 letters + 1 BOS)

---

## Data Rules

- names.txt is downloaded from:
  `https://raw.githubusercontent.com/karpathy/makemore/master/names.txt`
- Fetched in memory, not saved to disk unless explicitly added later
- Lines are trimmed and empty lines are dropped
- Names are lowercased (they already are in the source, but enforce it)
- Slice is shuffled after loading using `rand.Shuffle`

---

## Math / matutil Rules

- softmax: numerically stable (subtract max before exp)
- rmsnorm: standard RMS normalization
- matVecMul: matrix-vector multiply, row-major layout
- outer: outer product of two vectors
- vecAdd, vecScale: in-place or returning new slice — be consistent, document which
- No silent dimension mismatches — panic with a descriptive message if dims don't match

---

## Build Order

Implement files in this order. Do not skip ahead.

1. `main.go` — stub only, calls `train.Run()`
2. `data.go` — `LoadNames(url string) ([]string, error)`
3. `tokenizer.go` — `Tokenizer` struct + `NewTokenizer` + `Encode` + `Decode`
4. `matutil.go` — all math primitives
5. `model.go` — `GPT` struct + `NewGPT`
6. `attention.go` — forward attention
7. `cache.go` — `Cache` struct for activations
8. `model.go` (append) — `ForwardSeq`
9. `grad_attention.go` — backward for attention
10. `model.go` (append) — `Backward`
11. `adam.go` — `Adam` struct + `Update`
12. `train.go` — training loop
13. `main.go` — wire everything together

---

## Portfolio Context

This project is intentionally dependency-free to demonstrate:
- Understanding of transformer internals (not just API usage)
- Systems-level Go (manual memory layout, goroutines for perf later)
- Ability to translate math-heavy Python to a typed systems language

Do not suggest shortcuts that hide the math. The manual implementation IS the point.