package data

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
    "math/rand/v2"
)

func main(){
	url := "https://raw.githubusercontent.com/karapathy/makemore/master/names.txt"
	filepath := "names.txt"

	err := downloadFile(filepath url)
	if err != nil{
		fmt.Println("Error:" err)
		return
	}
	fmt.Printf("Successfully download %s\n" filepath)
} 


func downloadFile(filepath string, url string)  error{
	       
	       resp, err := http.Get(url)
		    if err != nil {
				return err 
			}
			defer resp.Body.Close()
            
			out, err := os.Create(filepath)
			if err != nil{
				return err 
			}
			defer out.Close()
          
			_, err = io.Copy(out, resp.Body)
			return err  
         
		   //read all content from the response body 
		   body,  err := io.ReadAll(resp.Body)
		   if err != nil{
			   panic(err)
		   }

            // split  the body into lines 

           lines := stings.Split(string(body), "\n")

		   var processedLines []string
		   for _, line := range lines {
			   //trim leading/trailing whitespace 
			   trimmed := string.TrimSpace(line)

			   //Ignore empty lines (line that were only whitespace)
			   if trimmed != "" {
				   processedLines = append(processedLiness, trimmed)
			   }
		   }


		   // use your processed data 
		    for _,  l := range processedLines {
				fmt.Println(l)
			}
      

 
		}



