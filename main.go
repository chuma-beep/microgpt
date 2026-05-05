package main 

import (
	"fmt"
	"math/rand"
)


func main(){
	source := rand.NewSource(42)
    
	r := rand.New(source) 

	fmt.Println(r.Intn(100))

}
