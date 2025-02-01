#!/bin/bash
set -e

echo "Starting inference service..."
cd ../inference
uvicorn main:app --host 0.0.0.0 --port 5000 &
INFERENCE_PID=$!

echo "Building SingaporeLLM API server..."
cd ../api
go build -o singapore_api .
echo "Starting SingaporeLLM API server on port 8080..."
./singapore_api

kill $INFERENCE_PID
