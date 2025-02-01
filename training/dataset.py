import json
import logging
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, data_file, tokenizer, max_length=512):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        with open(data_file, "r") as f:
            for line in f:
                obj = json.loads(line)
                text = obj.get("text", "")
                if text:
                    self.samples.append(text)
        logging.info(f"Loaded {len(self.samples)} samples from {data_file}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text = self.samples[idx]
        encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded["attention_mask"].squeeze(0)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}
