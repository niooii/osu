import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import osu.dataset as dataset


class OsuReplayRNN:
    def __init__(self, batch_size=64, device=None, noise_std=0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)
        self.output_size = len(dataset.OUTPUT_FEATURES)
        self.noise_std = noise_std
        
        # Initialize model
        self.model = self._create_model(self.input_size, self.output_size, noise_std)
        self.model.to(self.device)
        
        # Initialize optimizer and loss
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.01)
        self.criterion = nn.SmoothL1Loss()
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        
        # Data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None
        
        print(f"RNN Model initialized on {self.device} (noise_std={noise_std})")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")
    
    def _create_model(self, input_size, output_size, noise_std):
        class OsuModel(nn.Module):
            def __init__(self, input_size, output_size, noise_std=0.0):
                super(OsuModel, self).__init__()
                
                self.lstm = nn.LSTM(input_size, 64, batch_first=True)
                self.dense1 = nn.Linear(64, 64)
                self.noise_std = noise_std
                self.dense2 = nn.Linear(64, 16)
                self.position = nn.Linear(16, output_size)
                
            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                
                pos = self.dense1(lstm_out)
                
                # Configurable gaussian noise (only applied during training)
                if self.training and self.noise_std > 0:
                    noise = torch.randn_like(pos) * self.noise_std
                    pos = pos + noise
                
                pos = self.dense2(pos)
                pos = self.position(pos)
                
                return pos
        
        return OsuModel(input_size, output_size, noise_std)
    
    def load_data(self, input_data, output_data, test_size=0.2):
        # Handle input/output data inconsistency by taking minimum
        min_samples = min(input_data.shape[0], output_data.shape[0])
        input_data = input_data[:min_samples]
        output_data = output_data[:min_samples]
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, output_data, test_size=test_size, random_state=42
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    def train(self, epochs: int):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            
            for batch_x, batch_y in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} (Train)'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)
                    
                    outputs = self.model(batch_x)
                    loss = self.criterion(outputs, batch_y)
                    epoch_test_loss += loss.item()
            
            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            avg_test_loss = epoch_test_loss / len(self.test_loader)
            
            self.train_losses.append(avg_train_loss)
            self.test_losses.append(avg_test_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')
    
    def plot_losses(self):
        plt.figure(figsize=(10, 5))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.test_losses, label='Test Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RNN Training Progress')
        plt.legend()
        plt.show()
    
    def save(self, path=None):
        if path is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'.trained/rnn_{timestamp}.pt'
        
        if not os.path.exists('.trained'):
            os.makedirs('.trained')
            
        torch.save(self.model.state_dict(), path)
        torch.save(self.model.state_dict(), '.trained/rnn_most_recent.pt')
        print(f"RNN model saved to {path}")
    
    @staticmethod
    def load(path, noise_std=0.0, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create instance without data
        rnn = OsuReplayRNN(device=device, noise_std=noise_std)
        rnn.model.load_state_dict(torch.load(path, map_location=device))
        rnn.model.eval()
        print(f"RNN model loaded from {path}")
        return rnn
    
    def generate(self, beatmap_data):
        self.model.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            return self.model(beatmap_tensor).cpu().numpy()