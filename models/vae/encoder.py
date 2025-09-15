import torch
import torch.nn as nn
import torch.nn.functional as F

# Global VAE configuration
LATENT_DIM = 48
FUTURE_FRAMES = 70
PAST_FRAMES = 20


class AttnPool(nn.Module):
    """
    Content-based attention pooling over a sequence.
    (B, T, H) -> (B, H)
    """
    def __init__(self, in_dim: int, hidden: int = 128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        # h: (B, T, H)
        w = self.score(h)                  # (B, T, 1)
        a = torch.softmax(w, dim=1)        # (B, T, 1)
        c = (a * h).sum(dim=1)             # (B, H)
        return c


class ReplayEncoder(nn.Module):
    def __init__(self, input_size, latent_dim=32, noise_std=0.0, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES, bidirectional: bool = False, use_windowing: bool = True):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames  # +1 for current frame
        self.bidirectional = bidirectional
        self.use_windowing = use_windowing

        # Windowed beatmap features + cursor positions
        # if use_windowing=False, we feed raw map features per frame
        # combined_size = (input_size * window_size) + 2  OR  (input_size) + 2
        combined_size = (input_size * self.window_size) + 2 if self.use_windowing else (input_size + 2)

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

        # attention pooling for no-windowing mode; pools sequence into a fixed-size vector
        self.attn_pool = AttnPool(in_dim=enc_out_dim, hidden=128)

        # Output layers for reparameterization trick
        self.mu_layer = nn.Linear(64, latent_dim)
        self.logvar_layer = nn.Linear(64, latent_dim)

    def forward(self, beatmap_features, positions):
        # beatmap_features is now already windowed
        # gaussian noise during training
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(positions) * self.noise_std
            positions = positions + noise

        # Combine beatmap features with positions per frame
        # (B, T, combined_size)
        x = torch.cat([beatmap_features, positions], dim=-1)

        # Encode sequence with LSTM
        # seq_out: (B, T, H), h_n: (layers*dirs, B, H/dir)
        seq_out, (h_n, _) = self.lstm(x)

        if self.use_windowing:
            # Original behavior: use last hidden state (keeps parity with old training)
            if self.bidirectional:
                # Take last layer's forward and backward states and concat → (B, 2H)
                h_forward = h_n[-2]
                h_backward = h_n[-1]
                h = torch.cat([h_forward, h_backward], dim=1)
            else:
                h = h_n[-1]
            p_embed = h                        # (B, H or 2H)
        else:
            # No-windowing mode: attention-pool over time for a global code
            p_embed = self.attn_pool(seq_out)  # (B, H or 2H)

        h = F.relu(self.dense1(p_embed))
        
        h = F.relu(self.dense2(h))

        # Output mean and log variance for reparameterization trick
        mu = self.mu_layer(h)
        logvar = self.logvar_layer(h)

        return mu, logvar
