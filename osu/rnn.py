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
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=0.001)
        self.pos_criterion = nn.SmoothL1Loss()
        self.key_criterion = nn.BCEWithLogitsLoss()

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
                self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
                self.dense1 = nn.Linear(128, 128)
                self.noise_std = noise_std
                self.dense2 = nn.Linear(128, 64)
                # erm so  this doens't really care about the output size at all
                # we just hardcode it because what can go wrong with
                # x, y, k1, k2
                self.position = nn.Linear(64, 2)
                self.keys = nn.Linear(64, 2)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)

                features = nn.functional.relu(self.dense1(lstm_out))

                # gaussian noise (only applied during training)
                if self.training and self.noise_std > 0:
                    noise = torch.randn_like(features) * self.noise_std
                    features = features + noise

                features = nn.functional.relu(self.dense2(features))

                pos = self.position(features)
                keys = self.keys(features)

                return pos, keys

        return OsuModel(input_size, output_size, noise_std)

    def load_data(self, input_data, output_data, test_size=0.2):
        # Handle input/output data inconsistency by taking minimum
        # min_samples = min(input_data.shape[0], output_data.shape[0])
        # input_data = input_data[:min_samples]
        # output_data = output_data[:min_samples]

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, output_data, test_size=test_size, random_state=42
        )

        # print(x_train)
        # print(y_train)

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    def train(self, epochs: int):
        if self.train_loader is None or self.test_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        def process_keys(keys_tensor: torch.FloatTensor):
            prev_states = torch.cat([
                torch.zeros(keys_tensor.shape[0], 1, 2, device=keys_tensor.device),
                keys_tensor[:, :-1, :]
            ], dim=1)
            # print(prev_states.shape)

            # [64, 2048, 2]
            presses = torch.logical_and(keys_tensor == 1, prev_states == 0)
            # [64, 2]
            press_counts = presses.sum(dim=1)
            # [64] bool tensor (swap values?)
            swap_mask = press_counts[:, 1] > press_counts[:, 0]  # k2 > k1

            keys_tensor[swap_mask] = torch.flip(keys_tensor[swap_mask], dims=[2])

        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0

            for batch_x, batch_y in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} (Train)'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # [batch_size, sequence_length, 2] (x, y)
                batch_y_pos, batch_y_keys = torch.split(batch_y, [2, 2], dim=2)

                # Find the dominant key. Players will usually singletap the key that their
                # dominant finger is on, which means that key will have a lot more clicks
                # (or slightly more, the difference will show)
                # we "normalize" the different dominant keys by finding the axis
                # with the most clicks (changes from 0 -> 1), and then rotate the array
                # such that the axis with the most clicks is always first.
                process_keys(batch_y_keys)

                self.optimizer.zero_grad()
                pos, keys = self.model(batch_x)

                pos_loss = self.pos_criterion(pos, batch_y_pos)
                keys_loss = self.key_criterion(keys, batch_y_keys)
                loss = pos_loss + keys_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_train_loss += loss.item()

            # Validation phase
            self.model.eval()
            epoch_test_loss = 0
            with torch.no_grad():
                for batch_x, batch_y in self.test_loader:
                    batch_x = batch_x.to(self.device)
                    batch_y = batch_y.to(self.device)

                    batch_y_pos, batch_y_keys = torch.split(batch_y, [2, 2], dim=2)

                    process_keys(batch_y_keys)

                    pos, keys = self.model(batch_x)
                    pos_loss = self.pos_criterion(pos, batch_y_pos)
                    keys_loss = self.key_criterion(keys, batch_y_keys)
                    loss = pos_loss + keys_loss
                    epoch_test_loss += loss.item()

            # Calculate average losses
            avg_train_loss = epoch_train_loss / len(self.train_loader)
            avg_test_loss = epoch_test_loss / len(self.test_loader)

            self.train_losses.append(avg_train_loss)
            self.test_losses.append(avg_test_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Test Loss: {avg_test_loss:.4f}')

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
        from osu.rulesets.keys import convert_single_key_to_dual

        self.model.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            pos, keys = self.model(beatmap_tensor)

            key_probs = torch.sigmoid(keys)
            keys = (key_probs > 0.5).float()

            # Combine into [x, y, k1, k2] format
            model_output = torch.cat([pos, keys], dim=-1).cpu().numpy()

            return model_output
