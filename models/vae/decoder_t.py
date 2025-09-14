import torch
import torch.nn as nn
import torch.nn.functional as F

from osu.dataset import SEQ_LEN

from ..model_utils import TransformerArgs


# A transformer variant of the old RNN decoder,
# It should be noted that the encoder already has a frame window when generating the map
# embeddings, so it is probably ok to set past_frames and future_frames to default (0)
# unless you want explicit access to older and future embeddings...? not implemented
# for now tho
# TODO! positional encodings
class ReplayDecoderT(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        transformer_args: TransformerArgs = None,
        seq_len: int = SEQ_LEN,
        past_frames: int = 0,
        future_frames: int = 0,
    ):
        super().__init__()

        self.transformer_args = transformer_args or TransformerArgs()
        self.latent_dim = latent_dim
        self.past_frames = past_frames
        self.future_frames = future_frames
        # the embed dim must be the dim of the map embeddings
        embed_dim = self.transformer_args.embed_dim

        self.pos_dec = nn.Parameter(
            torch.randn(seq_len, self.transformer_args.embed_dim)
        )

        self.pos_head = nn.Sequential(
            nn.Linear(in_features=embed_dim, out_features=embed_dim // 2),
            nn.ReLU(),
            nn.Linear(in_features=embed_dim // 2, out_features=2),
        )

        self.proj_tgt = nn.Linear(
            in_features=embed_dim + latent_dim, out_features=embed_dim
        )

        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=self.transformer_args.attn_heads,
            dim_feedforward=self.transformer_args.ff_dim,
            batch_first=True,
        )

        self.decoder = nn.TransformerDecoder(
            decoder_layer=self.decoder_layer,
            num_layers=self.transformer_args.transformer_layers,
        )

    def forward(self, map_embeddings, latent_code):
        batch_size, seq_len, _ = map_embeddings.shape

        # latent_code = (batch_size, latent_dim)

        # Expand latent code to sequence length
        # (batch_size, seq_len, latent_dim)
        # TODO think on this choice, why use diff latent vector for each chunk??
        latent_expanded = latent_code.unsqueeze(1).expand(-1, seq_len, -1)

        x = torch.cat([map_embeddings + self.pos_dec.unsqueeze(0), latent_expanded], dim=-1)
        # project back to embed_dim
        x = self.proj_tgt(x)

        decoded = self.decoder(tgt=x, memory=map_embeddings)

        pos_out = self.pos_head(decoded)

        return pos_out

    def to_dict(self):
        return {
            "transformer_args": self.transformer_args.to_dict(),
            "latent_dim": self.latent_dim,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
        }

    @classmethod
    def from_dict(cls, data):
        transformer_args = TransformerArgs.from_dict(data["transformer_args"])
        return cls(
            latent_dim=data["latent_dim"],
            transformer_args=transformer_args,
            past_frames=data.get("past_frames", 0),
            future_frames=data.get("future_frames", 0),
        )
