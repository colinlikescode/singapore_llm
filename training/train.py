import os
import math
import logging
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, get_linear_schedule_with_warmup, AdamW
from model import SingaporeLLM
from s3_helper import download_file, upload_model
from dataset import TextDataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_distributed():
    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    if distributed:
        torch.distributed.init_process_group(backend="nccl")
    return distributed

def train():
    epochs = int(os.getenv("EPOCHS", "10"))
    learning_rate = float(os.getenv("LEARNING_RATE", "0.0001"))
    bucket = os.getenv("S3_BUCKET", "singapore-llm-models")
    region = os.getenv("AWS_REGION", "us-east-1")
    model_key = os.getenv("MODEL_S3_KEY", "models/singapore_llm.pt")
    data_url = os.getenv("DATA_URL", "s3://prod-data-singapore-llm/training-data/dataset_v1.jsonl")
    tokenizer_model = os.getenv("TOKENIZER_MODEL", "bert-base-uncased")
    local_data_file = "dataset_v1.jsonl"
    checkpoint_dir = os.getenv("CHECKPOINT_DIR", "checkpoints")
    gradient_accumulation_steps = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "16"))
    save_steps = int(os.getenv("SAVE_STEPS", "1000"))
    batch_size = int(os.getenv("BATCH_SIZE", "8"))

    os.makedirs(checkpoint_dir, exist_ok=True)


    if not os.path.exists(local_data_file):
        logging.info("Downloading training data from %s", data_url)
        download_file(data_url, local_data_file, region)


    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
    dataset = TextDataset(local_data_file, tokenizer, max_length=512)

  
    distributed = setup_distributed()
    sampler = DistributedSampler(dataset) if distributed else None
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=(sampler is None), sampler=sampler)

 
    model = SingaporeLLM(vocab_size=tokenizer.vocab_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[torch.cuda.current_device()])


    optimizer = AdamW(model.parameters(), lr=learning_rate)
    total_steps = math.ceil(len(dataloader) / gradient_accumulation_steps) * epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(total_steps * 0.1), num_training_steps=total_steps)

 
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

    global_step = 0
    model.train()
    for epoch in range(epochs):
        if distributed:
            sampler.set_epoch(epoch)
        for step, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

      
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask)
                    loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                                               labels.view(-1),
                                                               ignore_index=tokenizer.pad_token_id)
                scaler.scale(loss).backward()
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
                loss = torch.nn.functional.cross_entropy(outputs.view(-1, outputs.size(-1)),
                                                           labels.view(-1),
                                                           ignore_index=tokenizer.pad_token_id)
                loss.backward()

      
            if (step + 1) % gradient_accumulation_steps == 0:
                if scaler:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1

       
                if global_step % save_steps == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"checkpoint-step-{global_step}.pt")
                    state_dict = model.module.state_dict() if distributed else model.state_dict()
                    torch.save(state_dict, ckpt_path)
                    logging.info("Checkpoint saved at step %d: %s", global_step, ckpt_path)

            if (step + 1) % 10 == 0:
                logging.info("Epoch [%d/%d] Step [%d/%d] Loss: %.4f", epoch+1, epochs, step+1, len(dataloader), loss.item())


    final_model_path = "singapore_llm.pt"
    state_dict = model.module.state_dict() if distributed else model.state_dict()
    torch.save(state_dict, final_model_path)
    logging.info("Final model saved locally as %s", final_model_path)
    upload_model(final_model_path, bucket, model_key, region)
    logging.info("Model uploaded to S3")

if __name__ == "__main__":
    train()
