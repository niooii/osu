
import torch
import torch.nn as nn
import torch.nn.functional as F

class ReplayDecoderT(nn.Module):
    def __init__(self, input_size, latent_dim=32):
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
