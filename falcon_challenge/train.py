import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformer import TransformerDecoder, H2Dataset, collate_fn, beam_search  # Import from transformer.py

# Define constants
DATA_PATH = 'data/h2/'  # Path to H2 data directory

# This script is designed to train the TransformerDecoder model on the H2 dataset.
# It includes data loading, model initialization, training loop, and evaluation.
# The model is trained on held-in data and can be fine-tuned on held-out data for few-shot adaptation.

# Define hyperparameters for the model and training
INPUT_DIM = 192  # Number of channels in neural data (2 arrays * 96 channels)
D_MODEL = 512  # Dimensionality of the model (embedding size)
NHEAD = 8  # Number of attention heads in the Transformer
NUM_ENCODER_LAYERS = 6  # Number of encoder layers
NUM_DECODER_LAYERS = 6  # Number of decoder layers
DIM_FEEDFORWARD = 2048  # Dimensionality of the feedforward network
VOCAB_SIZE = 100  # Size of the vocabulary (adjust based on actual vocab)
MAX_LEN = 1000  # Maximum sequence length
DROPOUT = 0.1  # Dropout rate
BATCH_SIZE = 32  # Batch size for training
NUM_EPOCHS = 100  # Number of training epochs
LEARNING_RATE = 1e-4  # Learning rate for the optimizer
WEIGHT_DECAY = 1e-5  # Weight decay for regularization
BEAM_WIDTH = 5  # Beam width for beam search during inference
MAX_DEC_LEN = 100  # Maximum length for generated sequences

# Define the vocabulary for H2 dataset
# This should match the unique characters in the cues
VOCAB = ['<pad>', '<sos>', '<eos>', '<unk>', ' ', '>', '.', '?'] + [chr(i) for i in range(ord('A'), ord('Z')+1)] + [chr(i) for i in range(ord('a'), ord('z')+1)]
VOCAB_SIZE = len(VOCAB)  # Update vocab size

def load_h2_data(data_path, split='held_in_calib'):
    """
    Load H2 data from NWB files.
    This function should be implemented to read from the actual NWB files in the data directory.
    For now, it returns dummy data.
    
    Args:
        data_path (str): Path to the data directory.
        split (str): Data split ('held_in_calib', 'held_in_eval', 'held_out_calib', etc.)
    
    Returns:
        neural_data (list): List of neural activity arrays [T, C]
        cues (list): List of cue strings
    """
    # Placeholder: In a real implementation, use pynwb to load from NWB files
    # Use data_path and split for actual loading
    print(f"Loading data from {data_path} for split {split}")
    # For demonstration, use dummy data
    neural_data = [torch.randn(100, INPUT_DIM), torch.randn(150, INPUT_DIM), torch.randn(120, INPUT_DIM)]
    cues = ["HELLO WORLD", "THIS IS A TEST", "ANOTHER EXAMPLE"]
    return neural_data, cues

def prepare_data(neural_data, cues, vocab):
    """
    Prepare the dataset and dataloader.
    
    Args:
        neural_data (list): List of neural arrays
        cues (list): List of cue strings
        vocab (list): Vocabulary list
    
    Returns:
        dataloader (DataLoader): DataLoader for the dataset
    """
    dataset = H2Dataset(neural_data, cues, vocab, max_len=MAX_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=0, shuffle=True)
    return dataloader

def train_model(model, dataloader, optimizer, criterion, num_epochs):
    """
    Train the TransformerDecoder model.
    
    Args:
        model (TransformerDecoder): The model to train
        dataloader (DataLoader): DataLoader for training data
        optimizer (torch.optim.Optimizer): Optimizer
        criterion (nn.Module): Loss function
        num_epochs (int): Number of epochs
    """
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for src, tgt in dataloader:
            optimizer.zero_grad()
            # Forward pass: src is [T, batch, C], tgt is [L, batch]
            output = model(src, tgt[:-1])  # Predict next tokens
            loss = criterion(output.view(-1, VOCAB_SIZE), tgt[1:].view(-1))  # Compute loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
        
        # Optional: Save model checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

def evaluate_model(model, dataloader, vocab):
    """
    Evaluate the model on validation data using beam search.
    
    Args:
        model (TransformerDecoder): The trained model
        dataloader (DataLoader): DataLoader for evaluation data
        vocab (list): Vocabulary list
    """
    model.eval()
    total_edit_distance = 0
    count = 0
    with torch.no_grad():
        for src, tgt in dataloader:
            # For each sample in batch
            for i in range(src.size(1)):  # batch dimension
                neural = src[:, i, :]  # [T, C]
                true_cue = tgt[:, i]  # [L]
                # Decode using beam search
                decoded = beam_search(model, neural, vocab, beam_width=BEAM_WIDTH, max_len=MAX_DEC_LEN)
                # Compute edit distance (placeholder)
                # In real implementation, use something like nltk.edit_distance
                true_str = ''.join([vocab[idx] for idx in true_cue if idx > 0])  # Skip <pad>
                edit_dist = len(decoded) - len(true_str)  # Dummy calculation
                total_edit_distance += edit_dist
                count += 1
    avg_edit_distance = total_edit_distance / count if count > 0 else 0
    print(f"Average Edit Distance: {avg_edit_distance:.4f}")

def main():
    """
    Main function to run the training and evaluation.
    """
    # Load training data
    print("Loading training data...")
    neural_data_train, cues_train = load_h2_data(DATA_PATH, split='held_in_calib')
    train_dataloader = prepare_data(neural_data_train, cues_train, VOCAB)
    
    # Load validation data
    print("Loading validation data...")
    neural_data_val, cues_val = load_h2_data(DATA_PATH, split='held_in_eval')
    val_dataloader = prepare_data(neural_data_val, cues_val, VOCAB)
    
    # Initialize model
    print("Initializing model...")
    model = TransformerDecoder(
        input_dim=INPUT_DIM,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        dim_feedforward=DIM_FEEDFORWARD,
        vocab_size=VOCAB_SIZE,
        max_len=MAX_LEN,
        dropout=DROPOUT
    )
    
    # Define optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore <pad> token
    
    # Train the model
    print("Starting training...")
    train_model(model, train_dataloader, optimizer, criterion, NUM_EPOCHS)
    
    # Evaluate the model
    print("Evaluating model...")
    evaluate_model(model, val_dataloader, VOCAB)
    
    # For held-out fine-tuning (few-shot)
    print("Loading held-out calibration data for fine-tuning...")
    neural_data_fewshot, cues_fewshot = load_h2_data(DATA_PATH, split='held_out_calib')
    fewshot_dataloader = prepare_data(neural_data_fewshot, cues_fewshot, VOCAB)
    
    # Fine-tune on few-shot data
    print("Fine-tuning on few-shot data...")
    train_model(model, fewshot_dataloader, optimizer, criterion, num_epochs=10)  # Fewer epochs for fine-tuning
    
    # Evaluate on held-out eval
    print("Evaluating on held-out data...")
    neural_data_heldout_eval, cues_heldout_eval = load_h2_data(DATA_PATH, split='held_out_eval')
    heldout_eval_dataloader = prepare_data(neural_data_heldout_eval, cues_heldout_eval, VOCAB)
    evaluate_model(model, heldout_eval_dataloader, VOCAB)

if __name__ == "__main__":
    main()
