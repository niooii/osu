
import torch
import torch.nn as nn
import torch.nn.functional as F

FUTURE_FRAMES = 70
PAST_FRAMES = 20


# A transformer variant of the old RNN decoder
class ReplayDecoderT(nn.Module):
    def __init__(self, input_size, latent_dim=32, past_frames=PAST_FRAMES, future_frames=FUTURE_FRAMES):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames

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
        pass
        # return positions
