import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import osu.dataset as dataset
from .annealer import Annealer
from .base import OsuModel

# Global VAE configuration
LATENT_DIM = 32
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


class OsuReplayVAE(OsuModel):
    def __init__(self, annealer: Annealer = None, batch_size=64, device=None, latent_dim=None, noise_std=0.0):
        # Store VAE-specific parameters before calling super().__init__
        self.latent_dim = latent_dim or LATENT_DIM
        self.noise_std = noise_std
        self.annealer = annealer or Annealer(
            total_steps=10,
            range=(0, 0.001),
            cyclical=True,
            stay_max_steps=5
        )
        
        # Call parent constructor which will call our abstract methods
        super().__init__(batch_size=batch_size, device=device, 
                        latent_dim=self.latent_dim, noise_std=noise_std, annealer=self.annealer)
    
    def _initialize_models(self, **kwargs):
        """Initialize VAE encoder and decoder models."""
        self.encoder = ReplayEncoder(self.input_size, self.latent_dim, self.noise_std,
                                     past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES)
        self.decoder = ReplayDecoder(self.input_size, self.latent_dim,
                                     past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES)
    
    def _initialize_optimizers(self):
        """Initialize VAE optimizer for both encoder and decoder."""
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.001)
    
    def _extract_target_data(self, output_data):
        """Extract position data (x, y) from output data for VAE training."""
        return output_data[:, :, :2]  # Take first 2 features (x, y)
    
    def _get_target_data_name(self):
        """Return description of target data type."""
        return "position only"

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

    def _train_epoch(self, epoch, total_epochs, **kwargs):
        """Train VAE for one epoch."""
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for batch_x, batch_y_pos in tqdm.tqdm(
                self.train_loader,
                desc=f'Epoch {epoch + 1}/{total_epochs} (Beta: {self.annealer.current()})'
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

        # Step the annealer
        self.annealer.step()

        return {
            'total_loss': avg_total_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }

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

    def _get_state_dict(self):
        """Get state dictionary for saving VAE model."""
        return {
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict(),
            'latent_dim': self.latent_dim,
            'input_size': self.input_size,
            'past_frames': PAST_FRAMES,
            'future_frames': FUTURE_FRAMES
        }
    
    def _load_state_dict(self, checkpoint):
        """Load state dictionary for VAE model."""
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])

    def generate(self, beatmap_data, num_samples=1, apply_bias_correction=True):
        """Generate position data - returns (x, y) positions"""
        self._set_eval_mode()

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

        self._set_train_mode()

        return pos.cpu().numpy()


    @classmethod  
    def load(cls, path, device=None, **kwargs):
        """Load VAE model from checkpoint."""
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(path, map_location=device)
        
        # Create instance with saved parameters
        latent_dim = checkpoint.get('latent_dim', LATENT_DIM)
        vae = cls(device=device, latent_dim=latent_dim, **kwargs)
        
        # Load state dict and set to eval mode
        vae._load_state_dict(checkpoint)
        # vae._set_eval_mode()
        print(f"VAE loaded from {path}")
        return vae
