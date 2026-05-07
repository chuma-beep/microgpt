// tokenizer.go – character level tokenizer with BOS
package main

import "sort"

type Tokenizer struct {
	CharToIdx map[byte]int
	IdxToChar []byte
	BOS       int
	VocabSize int
}

func NewTokenizer(names []string) *Tokenizer {
	// collect unique characters
	charSet := make(map[byte]bool)
	for _, name := range names {
		for i := 0; i < len(name); i++ {
			charSet[name[i]] = true
		}
	}
	// sort for reproducibility
	unique := make([]byte, 0, len(charSet))
	for c := range charSet {
		unique = append(unique, c)
	}
	sort.Slice(unique, func(i, j int) bool { return unique[i] < unique[j] })
	// build maps
	charToIdx := make(map[byte]int)
	idxToChar := make([]byte, len(unique))
	for i, c := range unique {
		charToIdx[c] = i
		idxToChar[i] = c
	}
	vocabSize := len(unique) + 1
	BOS := len(unique)
	return &Tokenizer{
		CharToIdx: charToIdx,
		IdxToChar: idxToChar,
		BOS:       BOS,
		VocabSize: vocabSize,
	}
}

func (t *Tokenizer) Encode(s string) []int {
	tokens := make([]int, 0, len(s)+2)
	tokens = append(tokens, t.BOS)
	for i := 0; i < len(s); i++ {
		tokens = append(tokens, t.CharToIdx[s[i]])
	}
	tokens = append(tokens, t.BOS)
	return tokens
}

func (t *Tokenizer) Decode(ids []int) string {
	var out []byte
	for _, id := range ids {
		if id == t.BOS {
			continue
		}
		out = append(out, t.IdxToChar[id])
	}
	return string(out)
}
