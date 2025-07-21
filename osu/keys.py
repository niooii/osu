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
from osu.model import KeyModel


class OsuKeyModel:
    def __init__(self, batch_size=64, device=None, noise_std=0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)
        self.noise_std = noise_std

        self.key_model = KeyModel(self.input_size, noise_std)
        self.key_model.to(self.device)
        
        if hasattr(torch, 'compile'):
            self.key_model = torch.compile(self.key_model)

        self.key_optimizer = optim.AdamW(self.key_model.parameters(), lr=0.01, weight_decay=0.001)
        self.key_criterion = nn.BCEWithLogitsLoss()

        # Training history
        self.train_losses = []
        self.test_losses = []

        # Data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None

        print(f"Key Model initialized on {self.device} (noise_std={noise_std})")
        key_params = sum(p.numel() for p in self.key_model.parameters() if p.requires_grad)
        print(f"Key model parameters: {key_params}")

    def load_data(self, input_data, output_data, test_size=0.2):
        """Load data - splits output_data internally to use only key data (k1, k2)"""
        # Extract only key data (k1, k2) from output_data
        key_data = output_data[:, :, 2:]  # Take last 2 features (k1, k2)
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, key_data, test_size=test_size, random_state=42
        )

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples (keys only)")

    def _process_keys(self, keys_tensor: torch.FloatTensor):
        """Process keys to normalize dominant key to first position"""
        prev_states = torch.cat([
            torch.zeros(keys_tensor.shape[0], 1, 2, device=keys_tensor.device),
            keys_tensor[:, :-1, :]
        ], dim=1)

        # [batch_size, seq_len, 2]
        presses = torch.logical_and(keys_tensor == 1, prev_states == 0)
        # [batch_size, 2]
        press_counts = presses.sum(dim=1)
        # [batch_size] bool tensor (swap values?)
        swap_mask = press_counts[:, 1] > press_counts[:, 0]  # k2 > k1

        keys_tensor[swap_mask] = torch.flip(keys_tensor[swap_mask], dims=[2])

    def train(self, epochs: int):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        for epoch in range(epochs):
            # Training phase
            self.key_model.train()
            epoch_train_loss = 0

            for batch_x, batch_y in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} (Keys)'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Process keys to normalize dominant key
                self._process_keys(batch_y)

                # Train key model
                self.key_optimizer.zero_grad()
                keys = self.key_model(batch_x)
                keys_loss = self.key_criterion(keys, batch_y)
                keys_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.key_model.parameters(), max_norm=1.0)
                self.key_optimizer.step()

                epoch_train_loss += keys_loss.item()

            # Validation phase
            self.key_model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    self._process_keys(batch_y)

                    keys = self.key_model(batch_x)
                    keys_loss = self.key_criterion(keys, batch_y)
                    epoch_test_loss += keys_loss.item()

            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            avg_test_loss = epoch_test_loss / len(self.test_loader)

            self.train_losses.append(avg_train_loss)
            self.test_losses.append(avg_test_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Train: {avg_train_loss:.4f}, Test: {avg_test_loss:.4f}')

    def plot_losses(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Keys')
        plt.plot(self.test_losses, label='Test Keys')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Key Model Training Loss')
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save(self, path=None):
        if path is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path = f'.trained/keys_{timestamp}.pt'

        if not os.path.exists('.trained'):
            os.makedirs('.trained')

        torch.save({
            'key_model': self.key_model.state_dict()
        }, path)
        torch.save({
            'key_model': self.key_model.state_dict()
        }, '.trained/keys_most_recent.pt')
        print(f"Key model saved to {path}")

    @staticmethod
    def load(path, noise_std=0.0, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        key_model = OsuKeyModel(device=device, noise_std=noise_std)
        
        checkpoint = torch.load(path, map_location=device)
        key_model.key_model.load_state_dict(checkpoint['key_model'])
        key_model.key_model.eval()
        print(f"Key model loaded from {path}")
        return key_model

    def generate(self, beatmap_data):
        """Generate key predictions - returns boolean array"""
        self.key_model.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            
            # Get key logits
            keys = self.key_model(beatmap_tensor)

            # Apply sigmoid and threshold to get boolean keys
            key_probs = torch.sigmoid(keys)
            keys_bool = (key_probs > 0.5).float()

            return keys_bool.cpu().numpy()