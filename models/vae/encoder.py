import torch
import torch.nn as nn
import torch.nn.functional as F

# Global VAE configuration
LATENT_DIM = 48
FUTURE_FRAMES = 70
PAST_FRAMES = 20


class ReplayEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=32, noise_std=0.0, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES, bidirectional: bool = False):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames  # +1 for current frame
        self.bidirectional = bidirectional

        # Windowed beatmap features + cursor positions
        combined_size = (input_size * self.window_size) + 2

        self.lstm = nn.LSTM(
            combined_size,
            128,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=self.bidirectional,
        )

        enc_out_dim = 128 * (2 if self.bidirectional else 1)
        self.dense1 = nn.Linear(enc_out_dim, 128)
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
        # x = beatmap_features

        # Encode sequence
        _, (h_n, _) = self.lstm(x)

        if self.bidirectional:
            # h_n shape: (num_layers * num_directions, B, hidden_size)
            # Take last layer's forward and backward states and concat
            h_forward = h_n[-2]
            h_backward = h_n[-1]
            h = torch.cat([h_forward, h_backward], dim=1)
        else:
            h = h_n[-1]

        h = F.relu(self.dense1(h))

        h = F.relu(self.dense2(h))

        # Output mean and log variance for reparameterization trick
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu, logvar
