package main

import (
	"log"
	"SingaporeLLM/api/handlers"

	"github.com/gin-gonic/gin"
)

func main() {
	router := gin.New()
	router.Use(gin.Logger())
	router.Use(gin.Recovery())

	// POST /infer endpoint to handle LLM token-based requests.
	router.POST("/infer", handlers.InferHandler)

	log.Println("Starting SingaporeLLM API server on port 8080")
	if err := router.Run(":8080"); err != nil {
		log.Fatalf("Server failed to start: %v", err)
	}
}
