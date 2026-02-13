import os
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# Load text from file
def load_text(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        return f.read()

# Preprocess text
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters but keep some punctuation if needed, 
    # for simplicity here we keep only alphanumeric and spaces similar to original but cleaner
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    words = text.split()
    return words

# Create sequences
def create_sequences(words, seq_length=10):
    sequences = []
    next_words = []
    for i in range(len(words) - seq_length):
        sequences.append(words[i:i + seq_length])
        next_words.append(words[i + seq_length])
    return sequences, next_words

# Prepare data
def prepare_data(sequences, next_words, word_to_idx):
    X = np.array([[word_to_idx[word] for word in seq] for seq in sequences])
    y = np.array([word_to_idx[word] for word in next_words])
    return X, y

# Custom Dataset class
class TextDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# RNN Model
class NextWordRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3, dropout=0.2):
        super(NextWordRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden, _) = self.lstm(embedded)
        # Take the output of the last time step
        output = self.dropout(lstm_out[:, -1, :]) 
        output = self.fc(output)
        return output

# Plot learning curves
def plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies):
    plt.figure(figsize=(15, 6))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join('rnn_model_results', 'learning_curves.png'))
    plt.close()

# Plot Confusion Matrix for Top N words
def plot_confusion_matrix_top_n(model, dataloader, idx_to_word, top_n=50):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(batch_y.cpu().numpy())
    
    # Convert to numpy arrays
    all_targets = np.array(all_targets)
    all_preds = np.array(all_preds)

    # Filter for top N most frequent words in targets
    unique_targets, counts = np.unique(all_targets, return_counts=True)
    sorted_indices = np.argsort(-counts)
    top_indices = unique_targets[sorted_indices[:top_n]]
    
    # Create mask for targets that are in top_indices
    mask = np.isin(all_targets, top_indices)
    
    # Apply mask
    filtered_targets = all_targets[mask]
    filtered_preds = all_preds[mask]
    
    labels = [idx_to_word[idx] for idx in top_indices]
    
    cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_indices)
    
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix (Top {top_n} Frequent Words in Validation Set)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join('rnn_model_results', 'confusion_matrix_top50.png'))
    plt.close()

# Calculate accuracy
def calculate_accuracy(outputs, targets):
    _, predicted = torch.max(outputs.data, 1)
    correct = (predicted == targets).sum().item()
    return correct / len(targets)

# Main execution
if __name__ == "__main__":
    # Create results dir
    os.makedirs('rnn_model_results', exist_ok=True)
    
    # Load and preprocess text
    print("Loading and preprocessing data...")
    if not os.path.exists("1661-0.txt"):
        print("Error: 1661-0.txt not found. Please download it first.")
        # Create a dummy file for testing if it doesn't exist? No, better warn user.
        exit(1)
        
    text = load_text("1661-0.txt")
    words = preprocess_text(text)
    
    # Create sequences
    seq_length = 20 # Increased sequence length for better context
    print(f"Creating sequences with length {seq_length}...")
    sequences, next_words = create_sequences(words, seq_length)
    
    # Create word mappings
    print("Building vocabulary...")
    word_counts = Counter(words)
    # Optional: Limit vocabulary size if it's too huge, but Sherlock Holmes should be fine (approx 8-10k words)
    vocab_size = len(word_counts)
    print(f"Vocabulary Size: {vocab_size}")
    
    # Sort words by frequency (most common first) - this helps our Top N logic later
    sorted_vocab = sorted(word_counts.items(), key=lambda item: item[1], reverse=True)
    word_to_idx = {word: idx for idx, (word, _) in enumerate(sorted_vocab)}
    idx_to_word = {idx: word for word, idx in word_to_idx.items()}
    
    # Prepare data
    X, y = prepare_data(sequences, next_words, word_to_idx)
    
    # Stratiy might fail if some classes have only 1 sample. 
    # Use simple random split.
    print("Splitting data...")
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create dataset and dataloader
    train_dataset = TextDataset(X_train, y_train)
    val_dataset = TextDataset(X_val, y_val)
    
    # Increased batch size for speed on GPU
    batch_size = 256 
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Initialize model
    print("Initializing model...")
    model = NextWordRNN(vocab_size, embedding_dim=128, hidden_dim=256, num_layers=3).to(device)
    
    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training parameters
    num_epochs = 20 # Reduced epochs as we have a larger model and better learning
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Training loop
    print(f"Starting training on {device}...")
    for epoch in range(num_epochs):
        # Training Phase
        model.train()
        total_train_loss = 0
        total_train_acc = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            total_train_acc += calculate_accuracy(outputs, batch_y)
            
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        
        # Validation Phase
        model.eval()
        total_val_loss = 0
        total_val_acc = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                
                total_val_loss += loss.item()
                total_val_acc += calculate_accuracy(outputs, batch_y)
                
        avg_val_loss = total_val_loss / len(val_loader)
        avg_val_acc = total_val_acc / len(val_loader)
        
        train_losses.append(avg_train_loss)
        train_accuracies.append(avg_train_acc)
        val_losses.append(avg_val_loss)
        val_accuracies.append(avg_val_acc)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] '
              f'Train Loss: {avg_train_loss:.4f} Acc: {avg_train_acc:.4f} | '
              f'Val Loss: {avg_val_loss:.4f} Acc: {avg_val_acc:.4f}')
    
    # Save model and results
    torch.save(model.state_dict(), os.path.join('rnn_model_results', 'next_word_prediction_model.pth'))
    
    # Plot learning curves
    print("Generating learning curves...")
    plot_learning_curves(train_losses, train_accuracies, val_losses, val_accuracies)
    
    # Plot confusion matrix
    print("Generating confusion matrix for top 50 words...")
    plot_confusion_matrix_top_n(model, val_loader, idx_to_word, top_n=50)
    
    print("Training completed.")
    print(f"Model saved to rnn_model_results/next_word_prediction_model.pth")
    print(f"Results saved to rnn_model_results/")