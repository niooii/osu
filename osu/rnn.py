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
from osu.model import PosModel


class OsuReplayRNN:
    def __init__(self, batch_size=64, device=None, noise_std=0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)
        self.output_size = len(dataset.OUTPUT_FEATURES)
        self.noise_std = noise_std

        self.pos_model = PosModel(self.input_size, noise_std)
        self.pos_model.to(self.device)
        
        # Compile model for faster training
        if hasattr(torch, 'compile'):
            self.pos_model = torch.compile(self.pos_model)
            print("Position model compiled with torch.compile")

        self.pos_optimizer = optim.AdamW(self.pos_model.parameters(), lr=0.005, weight_decay=0.001)
        self.pos_criterion = nn.SmoothL1Loss()

        # history
        self.train_losses = []
        self.test_losses = []

        # data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None

        print(f"RNN Model initialized on {self.device} (noise_std={noise_std})")
        pos_params = sum(p.numel() for p in self.pos_model.parameters() if p.requires_grad)
        print(f"Position model parameters: {pos_params}")


    def load_data(self, input_data, output_data, test_size=0.2):
        """Load data - splits output_data internally to use only position data (x, y)"""
        # Extract only position data (x, y) from output_data
        position_data = output_data[:, :, :2]  # Take first 2 features (x, y)
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, position_data, test_size=test_size, random_state=42
        )

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples (position only)")

    def train(self, epochs: int):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        for epoch in range(epochs):
            # Training phase
            self.pos_model.train()
            epoch_train_loss = 0

            for batch_x, batch_y_pos in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} (Train)'):
                batch_x = batch_x.to(self.device)
                batch_y_pos = batch_y_pos.to(self.device)

                # Train position model
                self.pos_optimizer.zero_grad()
                pos = self.pos_model(batch_x)
                pos_loss = self.pos_criterion(pos, batch_y_pos)
                pos_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.pos_model.parameters(), max_norm=1.0)
                self.pos_optimizer.step()

                epoch_train_loss += pos_loss.item()

            # Validation phase
            self.pos_model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y_pos in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y_pos = batch_y_pos.to(self.device)

                    pos = self.pos_model(batch_x)
                    pos_loss = self.pos_criterion(pos, batch_y_pos)
                    epoch_test_loss += pos_loss.item()

            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            avg_test_loss = epoch_test_loss / len(self.test_loader)

            self.train_losses.append(avg_train_loss)
            self.test_losses.append(avg_test_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Train: {avg_train_loss:.4f}, Test: {avg_test_loss:.4f}')

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Position')
        plt.plot(self.test_losses, label='Test Position')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RNN Position Model Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save(self, path=None):
        if path is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'.trained/rnn_{timestamp}.pt'

        if not os.path.exists('.trained'):
            os.makedirs('.trained')

        # Save only position model (maintain dict structure for backwards compatibility)
        torch.save({
            'pos_model': self.pos_model.state_dict()
        }, path)
        torch.save({
            'pos_model': self.pos_model.state_dict()
        }, '.trained/rnn_most_recent.pt')
        print(f"RNN model saved to {path}")

    @staticmethod
    def load(path, noise_std=0.0, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create instance without data
        rnn = OsuReplayRNN(device=device, noise_std=noise_std)
        
        # Load position model
        checkpoint = torch.load(path, map_location=device)
        rnn.pos_model.load_state_dict(checkpoint['pos_model'])
        rnn.pos_model.eval()
        print(f"RNN model loaded from {path}")
        return rnn

    def generate(self, beatmap_data):
        """Generate position data only - returns (x, y) positions"""
        self.pos_model.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            
            # Get position output only
            pos = self.pos_model(beatmap_tensor)

            return pos.cpu().numpy()
