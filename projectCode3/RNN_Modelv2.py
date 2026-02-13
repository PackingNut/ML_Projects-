import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import logging
from tqdm import tqdm
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text data"""
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.long)
        self.targets = torch.tensor(targets, dtype=torch.long)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class NextWordRNN(nn.Module):
    """ Vanilla RNN model for next word prediction"""
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.3):
        super(NextWordRNN, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # RNN Model
        self.rnn = nn.RNN(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh',
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)    
        rnn_out, _ = self.rnn(embedded)       
        last_timestep = rnn_out[:, -1, :]      
        output = self.dropout(last_timestep)
        output = self.fc(output)              
        return output


def load_and_preprocess_text(filename, min_freq=2):
    """Load and preprocess text data"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            text = f.read()
        logger.info(f"Loaded text file with {len(text)} characters")
    except FileNotFoundError:
        logger.error(f"File {filename} not found")
        raise
    except Exception as e:
        logger.error(f"Error reading file {filename}: {e}")
        raise
    
    # tokenization
    words = text.lower().split()
    
    # Create vocabulary
    word_freq = Counter(words)
    word_to_idx = {'<PAD>': 0, '<UNK>': 1}
    
    for word, freq in word_freq.items():
        if freq >= min_freq:
            word_to_idx[word] = len(word_to_idx)
    
    logger.info(f"Vocabulary size: {len(word_to_idx)}")
    
    # Convert words to indices
    indices = [word_to_idx.get(word, 1) for word in words]
    
    return indices, word_to_idx


def create_sequences(indices, seq_length=15):
    """Create sequences for training"""
    sequences = []
    targets = []
    
    for i in range(len(indices) - seq_length):
        seq = indices[i:i + seq_length]
        target = indices[i + seq_length]
        sequences.append(seq)
        targets.append(target)
    
    return sequences, targets


def train_model(model, train_loader, val_loader, num_epochs=15, lr=0.002, device='cpu'):
    """Train the model with proper validation"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 3
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for batch_x, batch_y in pbar:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_train_acc += (predicted == batch_y).sum().item()
            
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        scheduler.step()
        
        # Validation phase
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val_acc += (predicted == batch_y).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader.dataset)
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader.dataset)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        logger.info(
            f'Epoch [{epoch+1}/{num_epochs}] '
            f'Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | '
            f'Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}'
        )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping triggered")
                break
    
    return train_losses, train_accuracies, val_losses, val_accuracies


def plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    """Plot training and validation curves"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_curves.png')
    plt.close()


def main():
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create results directory
    os.makedirs('rnn_results', exist_ok=True)
    
    # Load and preprocess data
    try:
        indices, word_to_idx = load_and_preprocess_text('1661-0.txt', min_freq=2)
        logger.info("Data preprocessing completed")
    except Exception as e:
        logger.error(f"Error in data preprocessing: {e}")
        return
    
    # Create sequences
    seq_length = 15
    sequences, targets = create_sequences(indices, seq_length)
    logger.info(f"Created {len(sequences)} sequences")
    
    # Split data
    split_idx = int(0.8 * len(sequences))
    train_seq = sequences[:split_idx]
    train_targets = targets[:split_idx]
    val_seq = sequences[split_idx:]
    val_targets = targets[split_idx:]
    
    # Create datasets and dataloaders
    train_dataset = TextDataset(train_seq, train_targets)
    val_dataset = TextDataset(val_seq, val_targets)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Create model
    vocab_size = len(word_to_idx)
    model = NextWordRNN(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2)
    model.to(device)
    
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    try:
        train_losses, train_accuracies, val_losses, val_accuracies = train_model(
            model, train_loader, val_loader, num_epochs=20, device=device
        )
        logger.info("Training completed successfully")
    except Exception as e:
        logger.error(f"Error during training: {e}")
        return
    
    # Plot results
    plot_training_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    logger.info("Training curves saved to training_curves.png")
    
    # Save vocabulary
    with open('word_to_idx.pkl', 'wb') as f:
        pickle.dump(word_to_idx, f)
    logger.info("Vocabulary saved to word_to_idx.pkl")
    
    # Save final model
    torch.save(model.state_dict(), 'final_model.pth')
    logger.info("Model saved to final_model.pth")


if __name__ == "__main__":
    main()
