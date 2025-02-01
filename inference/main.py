from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
import torch
from transformers import AutoTokenizer
from model import SingaporeLLM
from s3_helper import download_file

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = FastAPI(title="SingaporeLLM Inference Service")

class InferenceRequest(BaseModel):
    input_text: str

class InferenceResponse(BaseModel):
    output: str

@app.on_event("startup")
def load_resources():
    global model, tokenizer
    bucket = os.getenv("S3_BUCKET", "singapore-llm-models")
    region = os.getenv("AWS_REGION", "us-east-1")
    model_key = os.getenv("MODEL_S3_KEY", "models/singapore_llm.pt")
    model_path = "singapore_llm.pt"
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "bert-base-uncased")
    
    # Download model weights from S3 using the provided URL if available.
    model_weights_url = os.getenv("MODEL_WEIGHTS_URL", "")
    if model_weights_url:
        download_file(model_weights_url, model_path, region)
    else:
        download_file(f"s3://{bucket}/{model_key}", model_path, region)
    
    model = SingaporeLLM(vocab_size=30522)
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        logging.error("Failed to load model state: %s", e)
        raise e
    model.eval()
    logging.info("Model loaded successfully")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        logging.error("Failed to load tokenizer: %s", e)
        raise e

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    try:
        tokens = tokenizer.encode(request.input_text, return_tensors="pt")
    except Exception as e:
        logging.error("Tokenization error: %s", e)
        raise HTTPException(status_code=400, detail="Tokenization failed")
    
    with torch.no_grad():
        outputs = model(tokens)
    predictions = outputs.argmax(dim=-1).squeeze().tolist()
    if not isinstance(predictions, list):
        predictions = [predictions]
    try:
        output_text = tokenizer.decode(predictions, skip_special_tokens=True)
    except Exception as e:
        logging.error("Decoding error: %s", e)
        output_text = " ".join(str(token) for token in predictions)
    return InferenceResponse(output=output_text)
