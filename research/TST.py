import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class FinancialTST(nn.Module):
    def __init__(self, num_features, num_classes, d_model=32):
        super(FinancialTST, self).__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Input projection layer
        self.input_projection = nn.Linear(num_features, d_model)
        
        # Transformer layers
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=8)
        self.transformer = nn.TransformerEncoder(self.transformer_layer, num_layers=3)
        
        # Classification head
        self.classifier = nn.Linear(d_model, num_classes)

    def create_positional_encoding(self, seq_length, d_model):
        position = torch.arange(seq_length).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pos_encoding = torch.zeros(seq_length, d_model)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding.unsqueeze(0)

    def forward(self, past_values, past_time_features):
        # print("Input past_values shape:", past_values.shape)
        # print("Input past_time_features shape:", past_time_features.shape)
        
        # Reshape inputs to match the expected shape
        batch_size, _, seq_length, num_features = past_values.shape
        past_values = past_values.squeeze(1).permute(0, 1, 2)  # [batch_size, seq_length, num_features]
        
        # print("Reshaped past_values shape:", past_values.shape)
        
        # Project input to d_model dimensions
        past_values = self.input_projection(past_values)
        
        # Create and add positional encoding
        positional_encoding = self.create_positional_encoding(seq_length, self.d_model).to(past_values.device)
        past_values += positional_encoding
        
        # Permute for transformer input [seq_length, batch_size, d_model]
        past_values = past_values.permute(1, 0, 2)
        
        # Pass through transformer
        transformer_output = self.transformer(past_values)
        
        # Use the last hidden state for classification
        last_hidden_state = transformer_output[-1, :, :]
        
        # Classify
        logits = self.classifier(last_hidden_state)
        return logits

class FinancialDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        sequence = self.data[idx:idx+self.seq_length]
        values = sequence[:, :-1]  # All features except the last (target)
        time_features = np.arange(self.seq_length).reshape(-1, 1)  # Use integer time steps
        target = sequence[-1, -1]  # Last value of the target column

        return {
            'past_values': torch.FloatTensor(values).unsqueeze(0),  # Shape: [1, seq_length, num_features]
            'past_time_features': torch.FloatTensor(time_features),  # Shape: [seq_length, 1]
            'target': torch.LongTensor([target])
        }

def prepare_data(df, features, target):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df[features + [target]])
    return scaled_data

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            past_values=batch['past_values'].to(device),
            past_time_features=batch['past_time_features'].to(device)
        )
        loss = criterion(outputs, batch['target'].squeeze().to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in val_loader:
            outputs = model(
                past_values=batch['past_values'].to(device),
                past_time_features=batch['past_time_features'].to(device)
            )
            loss = criterion(outputs, batch['target'].squeeze().to(device))
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += batch['target'].size(0)
            correct += (predicted == batch['target'].squeeze().to(device)).sum().item()
    accuracy = correct / total
    return total_loss / len(val_loader), accuracy

def fine_tune_tst(df, features, target, seq_length, batch_size=16, num_epochs=10, learning_rate=0.001):
    data = prepare_data(df, features, target)
    train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

    train_dataset = FinancialDataset(train_data, seq_length)
    val_dataset = FinancialDataset(val_data, seq_length)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FinancialTST(len(features), num_classes=3).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = validate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    return model

# Usage
if __name__ == "__main__":
    df = pd.read_csv('market_df.csv')
    features = ["Close", "Open", "Volume", "High", "Low", "conversion_line", "base_line", "span_a", "span_b", "EMA", "ADX", "+DI", "-DI", "Momentum", "ROC", "CCI", "MACD", "MACD_signal", "MACD_hist", "RSI", "Stoch_k", "Stoch_d", "OBV", "ADL", "Upper_BB", "Middle_BB", "Lower_BB", "ATR", 'MA50', 'MA200', "Weighted_Close", "TWAP", "VWAP"]
    target = 'label'  # Your target column name
    seq_length = 10  # Adjust based on your needs

    model = fine_tune_tst(df, features, target, seq_length)
    torch.save(model.state_dict(), 'fine_tuned_tst.pth')