package main

import (
	"testing"
)

func TestTokenizerNew(t *testing.T) {
	names := []string{"abc", "def", "hello"}
	tok := NewTokenizer(names)

	// Should have 'a','b','c','d','e','f','h','l','o' = 9 unique chars + 1 BOS = 10
	if tok.VocabSize != 10 {
		t.Errorf("VocabSize = %d, want 10", tok.VocabSize)
	}
	if tok.BOS != 9 {
		t.Errorf("BOS = %d, want 9", tok.BOS)
	}

	// Check BOS maps to the last index
	if len(tok.IdxToChar) != 9 {
		t.Errorf("len(IdxToChar) = %d, want 9 (chars only, no BOS)", len(tok.IdxToChar))
	}

	// Verify characters are sorted
	for i := 1; i < len(tok.IdxToChar); i++ {
		if tok.IdxToChar[i] < tok.IdxToChar[i-1] {
			t.Errorf("IdxToChar not sorted: '%c' before '%c'", tok.IdxToChar[i-1], tok.IdxToChar[i])
		}
	}
}

func TestEncodeDecodeRoundtrip(t *testing.T) {
	names := []string{"alice", "bob", "charlie"}
	tok := NewTokenizer(names)

	tests := []string{"alice", "bob", "charlie"}
	for _, name := range tests {
		encoded := tok.Encode(name)
		decoded := tok.Decode(encoded)
		if decoded != name {
			t.Errorf("roundtrip(%q) = %q, want %q", name, decoded, name)
		}
	}
}

func TestEncodeBOSWrapping(t *testing.T) {
	tok := NewTokenizer([]string{"ab"})
	encoded := tok.Encode("ab")

	// Format should be [BOS, 'a', 'b', BOS]
	if len(encoded) != 4 {
		t.Fatalf("Encode('ab') len = %d, want 4", len(encoded))
	}
	if encoded[0] != tok.BOS || encoded[len(encoded)-1] != tok.BOS {
		t.Errorf("Encode('ab') = %v, BOS=%d, expected BOS at start and end", encoded, tok.BOS)
	}
}

func TestDecodeSkipsBOS(t *testing.T) {
	tok := NewTokenizer([]string{"xy"})
	encoded := tok.Encode("xy") // [BOS, x, y, BOS]
	decoded := tok.Decode(encoded)
	if decoded != "xy" {
		t.Errorf("Decode = %q, want 'xy'", decoded)
	}
	if len(decoded) != 2 {
		t.Errorf("Decoded length = %d, want 2", len(decoded))
	}
}

func TestTokenizerSingleChar(t *testing.T) {
	tok := NewTokenizer([]string{"a"})
	encoded := tok.Encode("a")
	if len(encoded) != 3 { // [BOS, a, BOS] = 3 tokens
		t.Errorf("Encode('a') len = %d, want 3", encoded)
	}
	decoded := tok.Decode(encoded)
	if decoded != "a" {
		t.Errorf("Decode = %q, want 'a'", decoded)
	}
}

func TestTokenizerEmptyNames(t *testing.T) {
	// Should handle empty name list gracefully (empty names filtered)
	tok := NewTokenizer([]string{""})
	if tok.VocabSize != 1 { // only BOS
		t.Errorf("VocabSize for empty names = %d, want 1 (BOS only)", tok.VocabSize)
	}
}
