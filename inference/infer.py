
import sys
import json
import os
import logging
import torch
from transformers import AutoTokenizer
from model import SingaporeLLM
from s3_helper import download_model

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_model():
    bucket = os.getenv("S3_BUCKET", "singapore-llm-models")
    region = os.getenv("AWS_REGION", "us-east-1")
    model_key = os.getenv("MODEL_S3_KEY", "models/singapore_llm.pt")
    model_path = "singapore_llm.pt"

    try:
        download_model(model_path, bucket, model_key, region)
    except Exception as e:
        logging.error("Failed to download model: %s", e)
        raise

    model = SingaporeLLM()
    try:
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except Exception as e:
        logging.error("Failed to load model state: %s", e)
        raise

    model.eval()
    logging.info("Model loaded successfully")
    return model

def run_inference(model, input_text):
    # Load a production-ready tokenizer (e.g., using a pretrained model)
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "bert-base-uncased")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    except Exception as e:
        logging.error("Failed to load tokenizer: %s", e)
        raise

    try:
        tokens = tokenizer.encode(input_text, return_tensors="pt")
    except Exception as e:
        logging.error("Tokenization error: %s", e)
        raise

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

    return output_text

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No input provided"}))
        sys.exit(1)

    input_text = sys.argv[1]
    try:
        model = load_model()
        output_text = run_inference(model, input_text)
        response = {"output": output_text}
    except Exception as e:
        logging.error("Inference failed: %s", e)
        response = {"error": "Inference failed"}
        print(json.dumps(response))
        sys.exit(1)

    print(json.dumps(response))

if __name__ == "__main__":
    main()
