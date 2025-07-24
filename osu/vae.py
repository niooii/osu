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

# Global VAE configuration
LATENT_DIM = 16
DEFAULT_BETA = 0.00085


class ReplayEncoder(nn.Module):
    """Encode replay sequence to latent representation"""
    def __init__(self, input_size, latent_dim=32):
        super().__init__()
        
        # Beatmap features + cursor positions
        combined_size = input_size + 2
        
        self.lstm = nn.LSTM(combined_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.mu_layer = nn.Linear(128, latent_dim)
        self.logvar_layer = nn.Linear(128, latent_dim)
        
    def forward(self, beatmap_features, positions):
        # Combine inputs
        x = torch.cat([beatmap_features, positions], dim=-1)
        
        # Encode sequence
        _, (h_n, _) = self.lstm(x)
        h = h_n[-1]  # Use last hidden state
        
        # Output mean and log variance for reparameterization trick
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)
        
        return mu, logvar


class ReplayDecoder(nn.Module):
    """Decode latent code + beatmap features to cursor positions"""
    def __init__(self, input_size, latent_dim=32):
        super().__init__()
        print("made newae")
        
        # beatmap features + latent code
        combined_size = input_size + latent_dim
        
        self.lstm = nn.LSTM(combined_size, 64, num_layers=2, batch_first=True)

        self.dense = nn.Linear(64, 32)

        self.output_layer = nn.Linear(32, 2)  # x, y positions
        
    def forward(self, beatmap_features, latent_code):
        batch_size, seq_len, _ = beatmap_features.shape
        
        # Expand latent code to sequence length
        latent_expanded = latent_code.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine inputs
        x = torch.cat([beatmap_features, latent_expanded], dim=-1)
        
        # Decode to positions
        lstm_out, _ = self.lstm(x)

        features = nn.functional.relu(self.dense(lstm_out))

        positions = self.output_layer(features)
        
        return positions


class OsuReplayVAE:
    def __init__(self, batch_size=64, device=None, latent_dim=None, beta=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.latent_dim = latent_dim or LATENT_DIM
        self.beta = beta or DEFAULT_BETA  # Weight for KL divergence
        self.input_size = len(dataset.INPUT_FEATURES)

        # Initialize models
        self.encoder = ReplayEncoder(self.input_size, self.latent_dim)
        self.decoder = ReplayDecoder(self.input_size, self.latent_dim)

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


        # Data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None

        print(f"VAE Models initialized on {self.device}")
        print(f"Encoder parameters: {sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)}")
        print(f"Decoder parameters: {sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)}")

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
        # Encode
        mu, logvar = self.encoder(beatmap_features, positions)
        
        # Sample latent code
        z = self.reparameterize(mu, logvar)
        
        # Decode
        reconstructed = self.decoder(beatmap_features, z)
        
        return reconstructed, mu, logvar

    def loss_function(self, reconstructed, original, mu, logvar):
        """VAE loss: reconstruction + KL divergence"""
        # Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original, reduction='mean')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kl_loss /= original.numel()  # Normalize by total elements
        
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss

    def train(self, epochs: int):
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_recon_loss = 0
            epoch_kl_loss = 0

            for batch_x, batch_y_pos in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} (VAE)'):
                batch_x = batch_x.to(self.device)
                batch_y_pos = batch_y_pos.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass
                reconstructed, mu, logvar = self.forward(batch_x, batch_y_pos)

                # Compute loss
                total_loss, recon_loss, kl_loss = self.loss_function(reconstructed, batch_y_pos, mu, logvar)

                # Backward pass
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(self.encoder.parameters()) + list(self.decoder.parameters()), max_norm=1.0)
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

            print(f'Epoch {epoch + 1}/{epochs}, Total: {avg_total_loss:.4f}, Recon: {avg_recon_loss:.4f}, KL: {avg_kl_loss:.4f}')

    def generate(self, beatmap_data, num_samples=1, apply_bias_correction=True):
        """Generate position data - returns (x, y) positions"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            batch_size = beatmap_tensor.shape[0]
            
            # Sample from prior distribution
            z = torch.randn(batch_size, self.latent_dim, device=self.device)
            
            # Generate positions
            pos = self.decoder(beatmap_tensor, z)
            
            # Apply bias correction if enabled and bias is stored
            if apply_bias_correction and hasattr(self, 'coordinate_bias'):
                pos = pos - torch.tensor(self.coordinate_bias, device=self.device)
            
        self.encoder.train()
        self.decoder.train()

        return pos.cpu().numpy()

    def compute_coordinate_bias(self, n_samples=100):
        """Compute coordinate bias by comparing generated vs training data"""
        if self.train_loader is None:
            print("No training data loaded")
            return
            
        self.encoder.eval()
        self.decoder.eval()
        
        training_positions = []
        generated_positions = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y_pos) in enumerate(self.train_loader):
                if i >= n_samples // self.batch_size:
                    break
                    
                batch_x = batch_x.to(self.device)
                batch_y_pos = batch_y_pos.to(self.device)
                
                # Store training positions
                training_positions.append(batch_y_pos.cpu().numpy())
                
                # Generate with random latent codes
                z = torch.randn(batch_x.shape[0], self.latent_dim, device=self.device)
                generated = self.decoder(batch_x, z)
                generated_positions.append(generated.cpu().numpy())
        
        # Compute means
        training_mean = np.concatenate(training_positions, axis=0).mean(axis=(0,1))
        generated_mean = np.concatenate(generated_positions, axis=0).mean(axis=(0,1))
        
        # Store bias for correction
        self.coordinate_bias = generated_mean - training_mean
        
        print(f"Coordinate bias computed: ({self.coordinate_bias[0]:.6f}, {self.coordinate_bias[1]:.6f})")
        
        self.encoder.train()
        self.decoder.train()
        
        return self.coordinate_bias

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
            'input_size': self.input_size
        }, f'{path_prefix}.pt')
        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'latent_dim': self.latent_dim,
            'input_size': self.input_size
        }, '.trained/vae_most_recent.pt')
        print(f"VAE models saved to {path_prefix}.pt")

    @staticmethod
    def load(path, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load checkpoint
        checkpoint = torch.load(path, map_location=device)
        
        # Create instance
        latent_dim = checkpoint.get('latent_dim', LATENT_DIM)
        vae = OsuReplayVAE(device=device, latent_dim=latent_dim)

        # Load models
        vae.encoder.load_state_dict(checkpoint['encoder'])
        vae.decoder.load_state_dict(checkpoint['decoder'])
        vae.encoder.train()
        vae.decoder.train()
        print(f"VAE models loaded from {path}")
        return vae

    def test_reconstruction(self):
        """Test how well the VAE reconstructs real data"""
        if self.test_loader is None:
            print("No test data available")
            return
            
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            # Get a test batch
            batch_x, batch_y_pos = next(iter(self.test_loader))
            batch_x = batch_x[:5].to(self.device)  # Use first 5 samples
            batch_y_pos = batch_y_pos[:5].to(self.device)
            
            # Reconstruct
            reconstructed, mu, logvar = self.forward(batch_x, batch_y_pos)
            
            # Calculate metrics
            mse = F.mse_loss(reconstructed, batch_y_pos).item()
            
            print(f"Reconstruction MSE: {mse:.6f}")
            print(f"Latent code statistics:")
            print(f"  Mean: {mu.mean().item():.6f}")
            print(f"  Std: {mu.std().item():.6f}")
            print(f"  LogVar mean: {logvar.mean().item():.6f}")
            
        self.encoder.train()
        self.decoder.train()

    def interpolate(self, beatmap_data, steps=10):
        """Interpolate between different latent codes"""
        self.encoder.eval()
        self.decoder.eval()
        
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            
            # Sample two random latent codes
            z1 = torch.randn(1, self.latent_dim, device=self.device)
            z2 = torch.randn(1, self.latent_dim, device=self.device)
            
            results = []
            for i in range(steps):
                alpha = i / (steps - 1)
                z_interp = (1 - alpha) * z1 + alpha * z2
                
                pos = self.decoder(beatmap_tensor, z_interp)
                results.append(pos.cpu().numpy())
                
        self.encoder.train()
        self.decoder.train()
        
        return np.concatenate(results, axis=0)