from dxcam.core import output
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from osu.dataset import BATCH_LENGTH
from models.base import OsuModel

FUTURE_FRAMES = 70
PAST_FRAMES = 20


# A transformer variant of the old RNN encoder
# TODO store hyperparam dict
class ReplayTransformer(OsuModel):
    def __init__(
        self, 
        input_size, 
        embed_dim=128,
        transformer_layers=4,
        ff_dim=1024,
        attn_heads=8,
        noise_std=0.0, 
        past_frames=PAST_FRAMES, 
        future_frames=FUTURE_FRAMES
    ):
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames  # +1 for current frame
        self.noise_std = noise_std

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.ff_dim = 1024
        self.attn_heads = 8
        
        super().__init__()

    def _initialize_models(self, **kwargs):
        # encoder step

        # project input features (9 or so -> embedding dim)
        self.proj_layer = nn.Linear(self.input_size, self.embed_dim)
        # learned position encodings (TODO! switch to rotating?)
        # BATCH_LENGTH is chunk lenght eg probably 2048
        self.pos_enc = nn.Parameter(torch.randn(BATCH_LENGTH, self.embed_dim))

        # consists of an attention layer followed by 2 linear layers: embed_dim -> ff_dim -> embed_dim
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.embed_dim, 
            nhead=self.attn_heads, 
            dim_feedforward=self.ff_dim, 
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=self.transformer_layers)

        # decoder step
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim, 
            nhead=self.attn_heads,
            dim_feedforward=self.ff_dim,
            batch_first=True
        )

        self.decoder = nn.TransformerDecoder(decoder_layer=self.decoder_layer, num_layers=self.transformer_layers)

        # unlike the other models, we learn [μx, μy, σx, σy] as
        # distribution parameters since we're no longer using latent codes
        self.pos_head = nn.Linear(self.embed_dim, 4)
        # we also learn keypresses because transformer big attention good
        self.key_head = nn.Linear(self.embed_dim, 2)

    def _initialize_optimizers(self):
        pass

    # maps beatmap features to their respective embeddings
    # (B, T, feature_dim) -> (B, T, embed_dim)
    # beatmap features is (B, T, feature_dim)
    def map_embeddings(self, beatmap_features):
        x = self.proj_layer(beatmap_features) # (B, T, embed_dim)

        # make it learn position encodings
        xp = x + self.pos_enc.unsqueeze(0)    # (B, T, embed_dim)
        
        # run it through the transformer layers
        # the mask gives a sliding window of context, allowing future frames to be attended to
        mask = self.local_mask(
            BATCH_LENGTH, 
            past_frames=self.past_frames, 
            future_frames=self.future_frames, 
        )

        return self.encoder(xp, mask=mask)    # (B, T, embed_dim)

    # override to get both positions and keypress data
    def _extract_target_data(self, output_data):
        return output_data


    def forward(self, beatmap_features, positions):
        # x = torch.cat([beatmap_features, positions], dim=-1)
        
        embeddings = self.map_embeddings(beatmap_features=beatmap_features)    # (B, T, embed_dim)

        self.decoder(memory=)

        # return mu, logvar

    # attention mask for the encoder, since global bidirectional attention isn't necessary
    def local_mask(self, T: int, past_frames: int, future_frames: int, device=None):
        device = device or torch.device('cuda')
        i = torch.arange(T, device=device).unsqueeze(1)     # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0)     # (1,T)
        allow = (j >= (i - past_frames)) & (j <= (i + future_frames))  # (T,T) bools
        mask = torch.full((T, T), float('-inf'), device=device)
        mask[allow] = 0.0
        return mask  # additive mask to be passed as src_mask
