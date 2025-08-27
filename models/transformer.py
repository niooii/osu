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
        self.proj_output_layer = nn.Linear(4, self.embed_dim)

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


    # output is (B, T, [x, y, k1, k2])
    def forward(self, beatmap_features, output):
        # x = torch.cat([beatmap_features, positions], dim=-1)
        
        # (B, T, embed_dim)
        map_embeddings = self.map_embeddings(beatmap_features=beatmap_features)    # (B, T, embed_dim)
        
        # (B, T - 1, 4)
        # [frame 1 ... frame 2047]
        dec_in = output[:, :-1]
        # [frame 2 ... frame 2048]
        tgt = output[:, 1:]

        # (B, T, embed_dim)
        play_embeddings = self.proj_output_layer(dec_in)

        # TODO! need rectnagular mask not square
        tgt_mask = self.local_mask(T=BATCH_LENGTH, past_frames=BATCH_LENGTH, future_frames=0)
        mem_mask = self.local_mask(T=BATCH_LENGTH, past_frames=90, future_frames=0)

        # (B, T, embed_dim)
        dec_out = self.decoder(tgt=play_embeddings, tgt_mask=tgt_mask, memory=map_embeddings, memory_mask=mem_mask)
        
        # (B, T, 4)
        pos_dist = self.pos_head(dec_out)
        # (B, T, 2)
        keys = self.key_head(dec_out)

        # return tgt for computing loss
        return pos_dist, keys, tgt

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        pass

    # attention mask since full global bidirectional attention isn't necessary
    def local_mask(self, T: int, past_frames: int, future_frames: int, device=None):
        device = device or torch.device('cuda')
        i = torch.arange(T, device=device).unsqueeze(1)     # (T,1)
        j = torch.arange(T, device=device).unsqueeze(0)     # (1,T)
        allow = (j >= (i - past_frames)) & (j <= (i + future_frames))  # (T,T) bools
        mask = torch.full((T, T), float('-inf'), device=device)
        mask[allow] = 0.0
        return mask  # additive mask to be passed as src_mask
