import torch
import torch.nn as nn
import torch.nn.functional as F

from osu.dataset import BATCH_LENGTH

from ..model_utils import TransformerArgs


# TODO store hyperparam dict
class ReplayEncoderT(nn.Module):
    def __init__(
        self,
        input_size,
        latent_dim: int,
        transformer_args: TransformerArgs = None,
        noise_std: float = 0.0,
        past_frames: int = 40,
        future_frames: int = 120,
    ):
        super().__init__()

        self.transformer_args = transformer_args or TransformerArgs()
        self.latent_dim = latent_dim
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.noise_std = noise_std

        # project input features (9 or so -> embedding dim)
        self.proj_layer = nn.Linear(input_size, self.transformer_args.embed_dim)
        # learned position encodings (TODO! switch to rotating?)
        # BATCH_LENGTH is chunk lenght eg probably 2048
        self.pos_enc = nn.Parameter(
            torch.randn(BATCH_LENGTH, self.transformer_args.embed_dim)
        )

        # consists of an attention layer followed by 2 linear layers: embed_dim -> ff_dim -> embed_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.transformer_args.embed_dim,
            nhead=self.transformer_args.attn_heads,
            dim_feedforward=self.transformer_args.ff_dim,
            batch_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=self.transformer_args.transformer_layers
        )

        self.mu_layer = nn.Sequential(
            nn.Linear(
                in_features=self.transformer_args.embed_dim, out_features=latent_dim * 2
            ),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim * 2, out_features=latent_dim),
        )

        self.logvar_layer = nn.Sequential(
            nn.Linear(
                in_features=self.transformer_args.embed_dim, out_features=latent_dim * 2
            ),
            nn.ReLU(),
            nn.Linear(in_features=latent_dim * 2, out_features=latent_dim),
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

    def forward(self, beatmap_features):
        # gaussian noise during training
        # if self.training and self.noise_std > 0:
        # noise = torch.randn_like(positions) * self.noise_std
        # positions = positions + noise

        # x = torch.cat([beatmap_features, positions], dim=-1)

        embeddings = self.map_embeddings(
            beatmap_features=beatmap_features
        )  # (B, T, embed_dim)

        # pool and compute mu and logvar for the latent code
        p_embed = embeddings.mean(dim=1)  # (B, embed_dim)

        mu = self.mu_layer(p_embed)
        logvar = self.logvar_layer(p_embed)

        return embeddings, mu, logvar

    # attention mask for the encoder, since global bidirectional attention isn't necessary
    def local_mask(self, T: int, past_frames: int, future_frames: int, device=None):
        device = device or torch.device("cuda")
        i = torch.arange(T, device=device).unsqueeze(1)  # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0)  # (1,T)
        allow = (j >= (i - past_frames)) & (j <= (i + future_frames))  # (T,T) bools
        mask = torch.full((T, T), float("-inf"), device=device)
        mask[allow] = 0.0
        return mask  # additive mask to be passed as src_mask

    def to_dict(self):
        return {
            "transformer_args": self.transformer_args.to_dict(),
            "latent_dim": self.latent_dim,
            "noise_std": self.noise_std,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
        }

    @classmethod
    def from_dict(cls, data, input_size):
        transformer_args = TransformerArgs.from_dict(data["transformer_args"])
        return cls(
            input_size=input_size,
            latent_dim=data["latent_dim"],
            transformer_args=transformer_args,
            noise_std=data.get("noise_std", 0.0),
            past_frames=data.get("past_frames", 0),
            future_frames=data.get("future_frames", 0),
        )
