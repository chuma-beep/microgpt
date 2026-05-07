// package main
//
// import (
// 	"fmt"
// )
//
// func main() {
// 	texts := []string{"hello", "world", "abc"}
// 	t := NewTokenizer(texts)
//
// 	fmt.Printf("Vocab size: %d\n", t.VocabSize())
// 	fmt.Printf("BOS: %d, UNK: %d\n", t.BOS, t.UNK)
// 	fmt.Printf("Index to char: %v\n", t.IdxToChar)
//
// 	encoded := t.Encode("hello")
// 	fmt.Printf("Encode 'hello': %v\n", encoded)
//
// 	decoded := t.Decode(encoded)
// 	fmt.Printf("Decode: %q\n", decoded)
//
// 	// test unknown char
// 	encoded2 := t.Encode("xyz")
// 	fmt.Printf("Encode 'xyz' (contains unknown): %v\n", encoded2)
// 	fmt.Printf("Decode: %q\n", t.Decode(encoded2))
// }
//
//

// main.go – entry point
package main

import (
	"flag"
)

func main() {
	steps := flag.Int("steps", 10000, "number of training steps")
	genTemp := flag.Float64("temperature", 0.5, "temperature for generation")
	weightsPath := flag.String("weights", "weights.bin", "path to weights file")
	flag.Parse()

	Run(*steps, *genTemp, *weightsPath)
}
