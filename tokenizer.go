package tokenizer

import (
	"sort"
)

// Tokenizer is a simple char-level tokenizer
type Tokenizer struct {
	CharToIdx map[rune]int
	IdxToChar []rune
	BOS       int
	UNK       int
}

// NewTokenizer builds vocab from dataset
func NewTokenizer(texts []string) *Tokenizer {
	vocabSet := make(map[rune]bool)

	// collect unique runes
	for _, text := range texts {
		for _, r := range text {
			vocabSet[r] = true
		}
	}

	// convert set → sorted slice (deterministic)
	var chars []rune
	for r := range vocabSet {
		chars = append(chars, r)
	}

	sort.Slice(chars, func(i, j int) bool {
		return chars[i] < chars[j]
	})

	// add special tokens at the front
	// index 0 = BOS, index 1 = UNK
	idxToChar := []rune{'^', '?'} // you can pick any symbols
	idxToChar = append(idxToChar, chars...)

	charToIdx := make(map[rune]int)
	for i, r := range idxToChar {
		charToIdx[r] = i
	}

	return &Tokenizer{
		CharToIdx: charToIdx,
		IdxToChar: idxToChar,
		BOS:       0,
		UNK:       1,
	}
}

// Encode converts string → token IDs
func (t *Tokenizer) Encode(text string) []int {
	tokens := make([]int, 0, len(text)+1)

	// prepend BOS
	tokens = append(tokens, t.BOS)

	for _, r := range text {
		if id, ok := t.CharToIdx[r]; ok {
			tokens = append(tokens, id)
		} else {
			tokens = append(tokens, t.UNK)
		}
	}

	return tokens
}

// Decode converts token IDs → string
func (t *Tokenizer) Decode(tokens []int) string {
	out := make([]rune, 0, len(tokens))

	for _, id := range tokens {
		// skip BOS
		if id == t.BOS {
			continue
		}

		if id >= 0 && id < len(t.IdxToChar) {
			out = append(out, t.IdxToChar[id])
		}
	}

	return string(out)
}

// VocabSize returns total number of tokens
func (t *Tokenizer) VocabSize() int {
	return len(t.IdxToChar)
}
