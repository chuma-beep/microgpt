// data.go – download and load names.txt
package main

import (
	"bufio"
	"io"
	"net/http"
	"strings"
)

func LoadNames(url string) ([]string, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	var names []string
	scanner := bufio.NewScanner(strings.NewReader(string(body)))
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}
		names = append(names, strings.ToLower(line))
	}
	return names, nil
}
