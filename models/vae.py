import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import osu.dataset as dataset
from .annealer import Annealer

# Global VAE configuration
LATENT_DIM = 48
FUTURE_FRAMES = 70
PAST_FRAMES = 20


class ReplayEncoder(nn.Module):
    """Encode replay sequence to latent representation"""

    def __init__(self, input_size, latent_dim=32, noise_std=0.0, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames  # +1 for current frame

        # Windowed beatmap features + cursor positions
        combined_size = (input_size * self.window_size) + 2

        self.lstm = nn.LSTM(combined_size, 128, num_layers=2, batch_first=True, dropout=0.2)

        self.dense1 = nn.Linear(128, 128)
        self.noise_std = noise_std
        self.dense2 = nn.Linear(128, 64)

        # Output layers for reparameterization trick
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

    def forward(self, beatmap_features, positions):
        # beatmap_features is now already windowed
        # gaussian noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(positions) * self.noise_std
            positions = positions + noise

        # Combine windowed beatmap features with positions
        x = torch.cat([beatmap_features, positions], dim=-1)

        # Encode sequence
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # Use last hidden state

        h = F.relu(self.dense1(h))

        h = F.relu(self.dense2(h))

        # Output mean and log variance for reparameterization trick
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu, logvar


# symmetric to the encoder
class ReplayDecoder(nn.Module):
    """Decode latent code + beatmap features to cursor positions"""

    def __init__(self, input_size, latent_dim=32, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames

        # Windowed beatmap features + latent code
        combined_size = (input_size * self.window_size) + latent_dim

        # Symmetric layers to encoder (not really almost)
        self.lstm = nn.LSTM(combined_size, 96, num_layers=2, batch_first=True, dropout=0.3)

        self.dense1 = nn.Linear(96, 96)
        self.dense2 = nn.Linear(96, 48)

        self.output_layer = nn.Linear(48, 2)  # x, y positions

    def forward(self, beatmap_features, latent_code):
        batch_size, seq_len, _ = beatmap_features.shape

        # beatmap_features is already windowed
        # Expand latent code to sequence length
        latent_expanded = latent_code.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine windowed features with latent code
        x = torch.cat([beatmap_features, latent_expanded], dim=-1)

        lstm_out, _ = self.lstm(x)

        features = F.relu(self.dense1(lstm_out))
        features = F.relu(self.dense2(features))

        positions = self.output_layer(features)

        return positions


class OsuReplayVAE:
    def __init__(self, annealer: Annealer = None, batch_size=64, device=None, latent_dim=None, noise_std=0.0):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.latent_dim = latent_dim or LATENT_DIM
        self.input_size = len(dataset.INPUT_FEATURES)
        self.annealer = annealer or Annealer(
            total_steps=10,
            range=(0, 0.001),
            cyclical=True,
            stay_max_steps=5
        )

        self.noise_std = noise_std
        self.encoder = ReplayEncoder(self.input_size, self.latent_dim, noise_std,
                                     past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES)
        self.decoder = ReplayDecoder(self.input_size, self.latent_dim,
                                     past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES)

        self.encoder.to(self.device)
        self.decoder.to(self.device)

        if hasattr(torch, 'compile'):
            self.encoder = torch.compile(self.encoder)
            self.decoder = torch.compile(self.decoder)
            print("VAE models compiled..")

        # Optimizer
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)

        # Training history
        self.total_losses = []
        self.recon_losses = []
        self.kl_losses = []

        self.train_loader = None
        self.test_loader = None

        print(f"VAE Models initialized on {self.device} (noise_std={noise_std})")
        print(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}")
        print(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}")

    def create_windowed_features(self, beatmap_features):
        """Convert sequential features to windowed features"""
        batch_size, seq_len, feat_size = beatmap_features.shape

        # Pad the sequence
        padded = F.pad(beatmap_features,
                       (0, 0, PAST_FRAMES, FUTURE_FRAMES, 0, 0),
                       mode='constant', value=0)

        # Create windows
        windows = []
        for i in range(seq_len):
            # Extract window: [past...current...future]
            window = padded[:, i:i + PAST_FRAMES + 1 + FUTURE_FRAMES, :]
            # Flatten the window
            window_flat = window.reshape(batch_size, -1)
            windows.append(window_flat)

        return torch.stack(windows, dim=1)

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

    def reparameterize(self, mu, logvar):
        """Reparameterization trick for backpropagation through sampling"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, beatmap_features, positions):
        """Forward pass through VAE"""
        # Create windowed features
        windowed_features = self.create_windowed_features(beatmap_features)

        # Encode with windowed features
        mu, logvar = self.encoder(windowed_features, positions)

        # Sample latent code
        z = self.reparameterize(mu, logvar)

        # Decode with windowed features
        reconstructed = self.decoder(windowed_features, z)

        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, original, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss (sum reduction for proper VAE training)
        recon_loss = F.mse_loss(reconstructed, original, reduction='sum')

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= original.shape[0]

        total_loss = recon_loss + self.annealer(kld)

        return total_loss, recon_loss, kld

    def train(self, epochs: int):
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0

            for batch_x, batch_y_pos in tqdm.tqdm(
                    self.train_loader,
                    desc=f'Epoch {epoch + 1}/{epochs} (Beta: {self.annealer.current()})'
            ):
                batch_x = batch_x.to(self.device)
                batch_y_pos = batch_y_pos.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                reconstructed, mu, logvar = self.forward(batch_x, batch_y_pos)

                # Compute loss
                total_loss, recon_loss, kl_loss = self.loss_function(reconstructed, batch_y_pos, mu, logvar)

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()),
                                               max_norm=1.0)
                self.optimizer.step()

                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()

            # Calculate average losses
            avg_total_loss = epoch_total_loss / len(self.train_loader)
            avg_recon_loss = epoch_recon_loss / len(self.train_loader)
            avg_kl_loss = epoch_kl_loss / len(self.train_loader)

            self.total_losses.append(avg_total_loss)
            self.recon_losses.append(avg_recon_loss)
            self.kl_losses.append(avg_kl_loss)

            print(
                f'Epoch {epoch + 1}/{epochs}, Total: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')

            self.annealer.step()

    def generate(self, beatmap_data, num_samples=1, apply_bias_correction=True):
        """Generate position data - returns (x, y) positions"""
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

            # Create windowed features for generation
            windowed_features = self.create_windowed_features(beatmap_tensor)

            batch_size = beatmap_tensor.shape[0]

            # Sample from prior distribution
            z = torch.randn(batch_size, self.latent_dim, device=self.device)

            # Generate positions with windowed features
            pos = self.decoder(windowed_features, z)

            # Apply bias correction if enabled and bias is stored
            if apply_bias_correction and hasattr(self, 'coordinate_bias'):
                pos = pos - torch.tensor(self.coordinate_bias, device=self.device)

        self.encoder.train()
        self.decoder.train()

        return pos.cpu().numpy()

    def plot_losses(self):
        plt.figure(figsize=(15, 5))

        # Total losses
        plt.subplot(1, 3, 1)
        plt.plot(self.total_losses, label='Total Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        # Reconstruction loss
        plt.subplot(1, 3, 2)
        plt.plot(self.recon_losses, label='Reconstruction Loss', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Reconstruction Loss')
        plt.legend()

        # KL divergence loss
        plt.subplot(1, 3, 3)
        plt.plot(self.kl_losses, label='KL Divergence', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('KL Divergence')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, path_prefix=None):
        if path_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path_prefix = f'.trained/vae_{timestamp}'

        if not os.path.exists('.trained'):
            os.makedirs('.trained')

        # Save both models
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'past_frames': PAST_FRAMES,
            'future_frames': FUTURE_FRAMES
        }, f'{path_prefix}.pt')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'past_frames': PAST_FRAMES,
            'future_frames': FUTURE_FRAMES
        }, '.trained/vae_most_recent.pt')
        print(f"VAE models saved to {path_prefix}.pt")

    @staticmethod
    def load(path, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)

        # Handle both compiled and non-compiled model state dicts
        def fix_state_dict(state_dict):
            """Remove '_orig_mod.' prefix from compiled model keys"""
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('_orig_mod.'):
                    new_state_dict[k[10:]] = v  # Remove '_orig_mod.' prefix
                else:
                    new_state_dict[k] = v
            return new_state_dict

        # Create instance
        latent_dim = checkpoint.get('latent_dim', LATENT_DIM)

        # check if sliding window model
        encoder_state = fix_state_dict(checkpoint['encoder'])
        lstm_weight_shape = encoder_state['lstm.weight_ih_l0'].shape
        expected_old_size = len(dataset.INPUT_FEATURES) + 2  # old architecture

        if 'past_frames' in checkpoint and 'future_frames' in checkpoint:
            # New windowed model
            past_frames = checkpoint['past_frames']
            future_frames = checkpoint['future_frames']
            vae = OsuReplayVAE(device=device, latent_dim=latent_dim)

            # Recreate models with correct windowing
            vae.encoder = ReplayEncoder(vae.input_size, vae.latent_dim, vae.noise_std,
                                        past_frames=past_frames, future_frames=future_frames)
            vae.decoder = ReplayDecoder(vae.input_size, vae.latent_dim,
                                        past_frames=past_frames, future_frames=future_frames)
        elif lstm_weight_shape[1] == expected_old_size:
            # Old model without windowing
            raise ValueError(
                "This appears to be an old model without windowing support. "
                "Please retrain the model with the new windowed architecture, "
                "or use the old version of the code to load this model."
            )
        else:
            # Unknown model format
            raise ValueError(f"Unknown model format. LSTM input size: {lstm_weight_shape[1]}")

        vae.encoder.to(device)
        vae.decoder.to(device)

        # Load models with fixed state dicts
        vae.encoder.load_state_dict(fix_state_dict(checkpoint['encoder']))
        vae.decoder.load_state_dict(fix_state_dict(checkpoint['decoder']))
        vae.encoder.train()
        vae.decoder.train()
        print(f"VAE models loaded from {path}")
        return vae
