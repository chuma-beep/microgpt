//go:build !js

// main.go – entry point
package main

import (
	"flag"
)

func main() {
	steps := flag.Int("steps", 10000, "number of training steps")
	genTemp := flag.Float64("temperature", 0.5, "temperature for generation")
	weightsPath := flag.String("weights", "weights.bin", "path to weights file")
	generateOnly := flag.Bool("generate", false, "skip training and generate names from saved weights")
	flag.Parse()

	Run(*steps, *genTemp, *weightsPath, *generateOnly)
}
