import torch
import torch.nn as nn
import torch.nn.functional as F


# A transformer variant of the old RNN decoder,
# It should be noted that the encoder already has a frame window when generating the map
# embeddings, so it is probably ok to set past_frames and future_frames to default (0)
# unless you want explicit access to older and future embeddings...? not implemented
# for now tho
class ReplayDecoderT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        transformer_layers: int,
        ff_dim: int,
        attn_heads: int,
        past_frames: int = 0,
        future_frames: int = 0,
    ):
        input_size = embed_dim + latent_dim
        self.pos_head = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=input_size // 2),
            nn.ReLU(),
            nn.Linear(in_features=input_size // 2, out_features=2),
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=input_size, nhead=attn_heads, dim_feedforward=ff_dim
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer, num_layers=transformer_layers
        )

        pass

    def forward(self, map_embeddings, latent_code):
        batch_size, seq_len, _ = map_embeddings.shape

        # latent_code = (batch_size, latent_dim)

        # Expand latent code to sequence length
        # (batch_size, seq_len, latent_dim)
        # TODO think on this choice, why use diff latent vector for each chunk??
        latent_expanded = latent_code.unsqueeze(1).expand(-1, seq_len, -1)

        x = torch.cat([map_embeddings, latent_expanded], dim=-1)
        decoded = self.decoder(x)

        pos_out = self.pos_head(decoded)

        return pos_out
