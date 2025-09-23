import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import math
from collections import Counter
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size, max_len=5000, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        self.input_embedding = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_encoder_layers)
        
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_decoder = PositionalEncoding(d_model, max_len)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_decoder_layers)
        
        self.output_layer = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, tgt):
        # tgt: [tgt_len, batch_size]
        
        # Encoder
        src_emb = self.input_embedding(src)  # [seq_len, batch_size, d_model]
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.encoder(src_emb)  # [seq_len, batch_size, d_model]
        
        # Decoder
        tgt_emb = self.decoder_embedding(tgt)  # [tgt_len, batch_size, d_model]
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)  # [tgt_len, batch_size, d_model]
        
        output = self.output_layer(output)  # [tgt_len, batch_size, vocab_size]
        return output
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def encode(self, src):
        src_emb = self.input_embedding(src)
        src_emb = self.pos_encoder(src_emb)
        src_emb = self.dropout(src_emb)
        memory = self.encoder(src_emb)
        return memory
    
    def decode(self, tgt, memory):
        tgt_emb = self.decoder_embedding(tgt)
        tgt_emb = self.pos_decoder(tgt_emb)
        tgt_emb = self.dropout(tgt_emb)
        tgt_mask = self.generate_square_subsequent_mask(tgt.size(0)).to(tgt.device)
        output = self.decoder(tgt_emb, memory, tgt_mask=tgt_mask)
        output = self.output_layer(output)
        return output

class H2Dataset(Dataset):
    def __init__(self, neural_data, cues, vocab, max_len=1000):
        self.neural_data = neural_data  # list of [T, C] arrays
        self.cues = cues  # list of strings
        self.vocab = vocab
        self.max_len = max_len
        self.char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        
    def __len__(self):
        return len(self.neural_data)
    
    def __getitem__(self, idx):
        neural = torch.tensor(self.neural_data[idx], dtype=torch.float32)  # [T, C]
        cue = self.cues[idx]
        cue_indices = [self.char_to_idx.get(c, self.char_to_idx['<unk>']) for c in cue]
        cue_tensor = torch.tensor(cue_indices, dtype=torch.long)
        return neural, cue_tensor

def collate_fn(batch):
    neurals, cues = zip(*batch)
    # Pad neurals to max length in batch
    max_t = max(n.shape[0] for n in neurals)
    padded_neurals = []
    for n in neurals:
        pad_len = max_t - n.shape[0]
        padded = F.pad(n, (0, 0, 0, pad_len))  # pad time dimension
        padded_neurals.append(padded)
    neurals = torch.stack(padded_neurals)  # [batch, T, C]
    neurals = neurals.transpose(0, 1)  # [T, batch, C]
    
    # Pad cues
    max_l = max(len(c) for c in cues)
    padded_cues = []
    for c in cues:
        pad_len = max_l - len(c)
        padded = F.pad(c, (0, pad_len), value=0)  # assuming 0 is <pad>
        padded_cues.append(padded)
    cues = torch.stack(padded_cues)  # [batch, L]
    cues = cues.transpose(0, 1)  # [L, batch]
    
    return neurals, cues

def beam_search(model, src, vocab, beam_width=5, max_len=100):
    model.eval()
    with torch.no_grad():
        memory = model.encode(src.unsqueeze(1))  # [T, 1, d_model]
        vocab_size = len(vocab)
        char_to_idx = {char: idx for idx, char in enumerate(vocab)}
        idx_to_char = {idx: char for char, idx in char_to_idx.items()}
        
        # Start with <sos>
        start_token = char_to_idx.get('<sos>', 0)
        beams = [(torch.tensor([start_token]), 0.0)]  # (sequence, score)
        
        for _ in range(max_len):
            new_beams = []
            for seq, score in beams:
                if seq[-1] == char_to_idx.get('<eos>', vocab_size-1):
                    new_beams.append((seq, score))
                    continue
                tgt = seq.unsqueeze(1)  # [len, 1]
                output = model.decode(tgt, memory)  # [len, 1, vocab]
                probs = F.softmax(output[-1], dim=-1).squeeze()  # [vocab]
                top_probs, top_indices = torch.topk(probs, beam_width)
                for prob, idx in zip(top_probs, top_indices):
                    new_seq = torch.cat([seq, idx.unsqueeze(0)])
                    new_score = score + prob.item()
                    new_beams.append((new_seq, new_score))
            beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        
        best_seq = beams[0][0]
        decoded = ''.join([idx_to_char[idx.item()] for idx in best_seq[1:] if idx.item() in idx_to_char])  # skip <sos>
        return decoded

# Example usage
if __name__ == "__main__":
    # Dummy vocab
    vocab = ['<pad>', '<sos>', '<eos>', '<unk>', ' ', '>', '.', '?'] + [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
    
    # Model params
    input_dim = 192  # 2*96 channels
    d_model = 512
    nhead = 8
    num_encoder_layers = 6
    num_decoder_layers = 6
    dim_feedforward = 2048
    vocab_size = len(vocab)
    
    model = TransformerDecoder(input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, vocab_size)
    
    # Dummy data
    neural_data = [torch.randn(100, 192), torch.randn(150, 192)]
    cues = ["HELLO WORLD", "TEST"]
    
    dataset = H2Dataset(neural_data, cues, vocab)
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn, num_workers=0)
    
    # Training loop (simplified)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore <pad>
    
    for epoch in range(10):
        for src, tgt in dataloader:
            optimizer.zero_grad()
            output = model(src, tgt[:-1])  # input tgt without last token
            loss = criterion(output.view(-1, vocab_size), tgt[1:].view(-1))  # target without first token
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")
    
    # Inference
    test_neural = torch.randn(200, 192)
    decoded = beam_search(model, test_neural, vocab)
    print("Decoded:", decoded)
