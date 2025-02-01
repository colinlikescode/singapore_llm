import torch
import torch.nn as nn

class SingaporeLLM(nn.Module):
    def __init__(self, vocab_size=30522, hidden_dim=768, num_layers=12, nhead=8):
        super(SingaporeLLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)  
        x = embedded.transpose(0, 1)           
        x = self.transformer_encoder(x)
        x = x.transpose(0, 1)
        logits = self.fc(x)
        return logits
