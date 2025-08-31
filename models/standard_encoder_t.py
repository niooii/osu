import torch
import torch.nn as nn
import torch.nn.functional as F

from osu.dataset import BATCH_LENGTH
from .model_utils import TransformerArgs


# an encoder purely for learning beatmap embeddings
class MapEncoder(nn.Module):
    def __init__(
        self,
        input_size,
        transformer_args: TransformerArgs,
        past_frames: int,
        future_frames: int,
    ):
        super().__init__()
        self.past_frames = past_frames
        self.future_frames = future_frames

        # project input features (9 or so -> embedding dim)
        self.proj_layer = nn.Linear(input_size, transformer_args.embed_dim)
        # learned position encodings (TODO! switch to rotating?)
        # BATCH_LENGTH is chunk lenght eg probably 2048
        self.pos_enc = nn.Parameter(torch.randn(BATCH_LENGTH, transformer_args.embed_dim))

        # consists of an attention layer followed by 2 linear layers: embed_dim -> ff_dim -> embed_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_args.embed_dim,
            nhead=transformer_args.attn_heads,
            dim_feedforward=transformer_args.ff_dim,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=transformer_args.transformer_layers
        )

    # maps beatmap features to their respective embeddings
    # (B, T, feature_dim) -> (B, T, embed_dim)
    # beatmap features is (B, T, feature_dim)
    def map_embeddings(self, beatmap_features):
        x = self.proj_layer(beatmap_features)  # (B, T, embed_dim)

        # make it learn position encodings
        xp = x + self.pos_enc.unsqueeze(0)  # (B, T, embed_dim)

        # run it through the transformer layers
        # the mask gives a sliding window of context, allowing future frames to be attended to
        mask = self.local_mask(
            BATCH_LENGTH,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
        )

        return self.encoder(xp, mask=mask)  # (B, T, embed_dim)

    def forward(self, beatmap_features, positions):
        embeddings = self.map_embeddings(
            beatmap_features=beatmap_features
        )  # (B, T, embed_dim)

        return embeddings

    # attention mask for the encoder, since global bidirectional attention isn't necessary
    def local_mask(self, T: int, past_frames: int, future_frames: int, device=None):
        device = device or torch.device("cuda")
        i = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
        allow = (j >= (i - past_frames)) & (j <= (i + future_frames))  # (T,T) bools
        mask = torch.full((T, T), float("-inf"), device=device)
        mask[allow] = 0.0
        return mask  # additive mask to be passed as src_mask
