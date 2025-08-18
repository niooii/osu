import torch
import torch.nn as nn
import torch.nn.functional as F

# Global VAE configuration
FUTURE_FRAMES = 70
PAST_FRAMES = 20


class ReplayGenerator(nn.Module):
    """Generator for AVAE - IDENTICAL architecture to decoder but takes additional noise"""

    def __init__(self, input_size, latent_dim=32, noise_dim=16, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames
        self.noise_dim = noise_dim
        self.latent_dim = latent_dim

        # Windowed beatmap features + latent code + noise
        combined_size = (input_size * self.window_size) + latent_dim + noise_dim

        # IDENTICAL architecture to ReplayDecoder (as per AVAE paper)
        self.lstm = nn.LSTM(combined_size, 96, num_layers=2, batch_first=True, dropout=0.3)

        self.dense1 = nn.Linear(96, 96)
        self.dense2 = nn.Linear(96, 48)

        self.output_layer = nn.Linear(48, 2)  # x, y positions

    def forward(self, beatmap_features, latent_code, noise=None):
        """
        Args:
            beatmap_features: windowed beatmap features
            latent_code: z from encoder
            noise: xi additional random noise (optional)
        """
        batch_size, seq_len, _ = beatmap_features.shape

        # Generate noise if not provided
        if noise is None:
            noise = torch.randn(batch_size, self.noise_dim, device=beatmap_features.device)

        # Expand both latent code and noise to sequence length
        latent_expanded = latent_code.unsqueeze(1).expand(-1, seq_len, -1)
        noise_expanded = noise.unsqueeze(1).expand(-1, seq_len, -1)

        # Combine windowed features with latent code and noise
        x = torch.cat([beatmap_features, latent_expanded, noise_expanded], dim=-1)

        lstm_out, _ = self.lstm(x)

        features = F.relu(self.dense1(lstm_out))
        features = F.relu(self.dense2(features))

        positions = self.output_layer(features)

        return positions