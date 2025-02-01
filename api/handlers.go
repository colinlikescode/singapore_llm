package handlers

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"log"
	"net/http"
	"time"

	"github.com/gin-gonic/gin"
)

// InferenceRequest represents the incoming request payload.
type InferenceRequest struct {
	InputTokens string `json:"input_tokens" binding:"required"`
}

// InferenceResponse represents the response from the inference service.
type InferenceResponse struct {
	OutputTokens string `json:"output_tokens"`
}

// InferHandler handles inference requests.
func InferHandler(c *gin.Context) {
	var req InferenceRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		log.Printf("Bad Request - invalid JSON or missing input_tokens: %v", err)
		c.JSON(http.StatusBadRequest, gin.H{"error": "Bad Request: invalid JSON payload or missing 'input_tokens'"})
		return
	}

	if req.InputTokens == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "Bad Request: 'input_tokens' cannot be empty"})
		return
	}

	// Prepare the request payload for the inference microservice.
	// (Note: the inference microservice expects the field "input_text".)
	payload := map[string]string{"input_text": req.InputTokens}
	requestPayload, err := json.Marshal(payload)
	if err != nil {
		log.Printf("Failed to marshal inference request: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal Server Error: failed to marshal request"})
		return
	}

	inferenceURL := "http://localhost:5000/infer" // Update as needed.
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Post(inferenceURL, "application/json", bytes.NewBuffer(requestPayload))
	if err != nil {
		log.Printf("Inference service unreachable: %v", err)
		c.JSON(http.StatusServiceUnavailable, gin.H{"error": "Service Unavailable: could not reach inference service"})
		return
	}
	defer resp.Body.Close()

	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		log.Printf("Failed to read inference response: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal Server Error: failed to read response from inference service"})
		return
	}

	if resp.StatusCode != http.StatusOK {
		log.Printf("Inference service error: %s", body)
		c.JSON(http.StatusBadGateway, gin.H{"error": "Bad Gateway: inference service returned error", "details": string(body)})
		return
	}

	var inferenceResp InferenceResponse
	if err := json.Unmarshal(body, &inferenceResp); err != nil {
		log.Printf("Failed to parse inference response: %v", err)
		c.JSON(http.StatusInternalServerError, gin.H{"error": "Internal Server Error: failed to parse inference response"})
		return
	}

	// Return the inference response with tokens.
	c.JSON(http.StatusOK, inferenceResp)
}
