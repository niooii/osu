import torch
import torch.nn as nn
import torch.nn.functional as F
from osu.dataset import BATCH_LENGTH
from models.base import OsuModel
from typing import Optional

FUTURE_FRAMES = 70
PAST_FRAMES = 20


# A transformer variant of the old RNN encoder
# TODO store hyperparam dict
class ReplayTransformer(nn.Module):
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
        super().__init__()
        
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.window_size = past_frames + 1 + future_frames  # +1 for current frame
        self.noise_std = noise_std

        self.input_size = input_size
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.ff_dim = ff_dim
        self.attn_heads = attn_heads
        
        self._initialize_layers()

    def _initialize_layers(self):
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

        T_dec = dec_in.shape[1]

        # (B, T, embed_dim)
        play_embeddings = self.proj_output_layer(dec_in)

        tgt_mask = self.local_mask(T=BATCH_LENGTH, past_frames=BATCH_LENGTH, future_frames=0)
        mem_mask = self.cross_attn_mask(T_dec=T_dec, T_enc=BATCH_LENGTH, past_frames=120)

        # (B, T, embed_dim)
        dec_out = self.decoder(tgt=play_embeddings, tgt_mask=tgt_mask, memory=map_embeddings, memory_mask=mem_mask)
        
        # (B, T, 4)
        pos_dist = self.pos_head(dec_out)
        # (B, T, 2)
        keys = self.key_head(dec_out)

        # return tgt for computing loss
        return pos_dist, keys, tgt


    # attention mask since full global bidirectional attention isn't necessary
    def local_mask(self, T: int, past_frames: int, future_frames: int, device=None):
        device = device or next(self.parameters()).device
        i = torch.arange(T, device=device).unsqueeze(1)     # (T, 1)
        j = torch.arange(T, device=device).unsqueeze(0)     # (1, T)
        allow = (j >= (i - past_frames)) & (j <= (i + future_frames))  # (T, T) bools
        mask = torch.full((T, T), float('-inf'), device=device)
        mask[allow] = 0.0
        return mask  # additive mask to be passed as src_mask

    def cross_attn_mask(self, T_dec: int, T_enc: int, past_frames=120, device=None):
        device = device or next(self.parameters()).device
        
        i = torch.arange(T_dec, device=device).unsqueeze(1)  # (T_dec, 1)
        j = torch.arange(T_enc, device=device).unsqueeze(0)  # (1, T_enc)
        
        # map decoder positions to encoder positions
        enc_positions = (i * T_enc / T_dec).long()
        
        # allow attending to [enc_pos - lookback, enc_pos]
        allow = (j >= (enc_positions - past_frames)) & (j <= enc_positions)
        
        mask = torch.full((T_dec, T_enc), float('-inf'), device=device)
        mask[allow] = 0.0
        return mask


class OsuReplayTransformer(OsuModel):
    def __init__(
        self, 
        embed_dim=128,
        transformer_layers=4,
        ff_dim=1024,
        attn_heads=8,
        noise_std=0.0, 
        past_frames=PAST_FRAMES, 
        future_frames=FUTURE_FRAMES,
        batch_size=64,
        device=None,
    ):
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.ff_dim = ff_dim
        self.attn_heads = attn_heads
        self.noise_std = noise_std
        self.past_frames = past_frames
        self.future_frames = future_frames
        
        super().__init__(batch_size=batch_size, device=device, compile=True) 

    def _initialize_models(self, **kwargs):
        self.transformer = ReplayTransformer(
            input_size=self.input_size,
            embed_dim=self.embed_dim,
            transformer_layers=self.transformer_layers,
            ff_dim=self.ff_dim,
            attn_heads=self.attn_heads,
            noise_std=self.noise_std,
            past_frames=self.past_frames,
            future_frames=self.future_frames
        )

    def _initialize_optimizers(self):
        self.optimizer = torch.optim.AdamW(
            self.transformer.parameters(),
            lr=1e-4,
            weight_decay=0.01,
            betas=(0.9, 0.98) 
        )

    # override to get both positions and keypress data
    def _extract_target_data(self, output_data):
        return output_data

    def forward(self, beatmap_features, output):
        return self.transformer(beatmap_features, output)

    def _train_epoch(self, epoch: int, total_epochs: int) -> dict:
        epoch_pos_loss = 0
        epoch_keys_loss = 0
        
        num_batches = len(self.train_loader)

        for i, (batch_x, batch_y) in enumerate(self.train_loader):
            self._set_custom_train_status(f"Batch {i}/{num_batches}")
            
            self.optimizer.zero_grad()
            pos_dist, keys, tgt = self.forward(beatmap_features=batch_x, output=batch_y)
            
            # (B, T, 2)
            # the mean predictions (basically position pred)
            mu_xy = pos_dist[:, :, :2]

            # (B, T, 2)
            # the variance predictions
            sig_xy = pos_dist[:, :, 2:]

            pos_loss = F.gaussian_nll_loss(input=mu_xy, target=tgt[:, :, :2], var=sig_xy, full=True, reduction="mean")
            key_loss = F.binary_cross_entropy_with_logits(input=keys, target=batch_y[:, :, 2:], reduction="mean")

            total_loss = pos_loss + key_loss
            total_loss.backward()

            torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
            self.optimizer.step()

            epoch_pos_loss += pos_loss.item()
            epoch_keys_loss += key_loss.item()
            
        return {
            "pos_loss": epoch_pos_loss / num_batches,
            "key_loss": epoch_keys_loss / num_batches
        }

    def _get_state_dict(self):
        return {
            'transformer': self.transformer.state_dict(),
            'embed_dim': self.embed_dim,
            'transformer_layers': self.transformer_layers,
            'ff_dim': self.ff_dim,
            'attn_heads': self.attn_heads,
            'noise_std': self.noise_std,
            'past_frames': self.past_frames,
            'future_frames': self.future_frames,
            'input_size': self.input_size
        }
    
    def _load_state_dict(self, checkpoint):
        self.transformer.load_state_dict(checkpoint['transformer'])

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        checkpoint = torch.load(path, map_location=device)
        
        # load hyperparameters from checkpoint
        transformer_args = {
            'embed_dim': checkpoint.get('embed_dim', 128),
            'transformer_layers': checkpoint.get('transformer_layers', 4),
            'ff_dim': checkpoint.get('ff_dim', 1024),
            'attn_heads': checkpoint.get('attn_heads', 8),
            'noise_std': checkpoint.get('noise_std', 0.0),
            'past_frames': checkpoint.get('past_frames', PAST_FRAMES),
            'future_frames': checkpoint.get('future_frames', FUTURE_FRAMES)
        }
        
        instance = cls(device=device, **kwargs, **transformer_args)
        instance._load_state_dict(checkpoint)
        instance._set_eval_mode()
        
        print(f"{cls.__name__} loaded from {path}")
        return instance

    def generate(self, beatmap_data, _, _a):
        return []
    # def generate(self, beatmap_data, temperature=1.0):
    #     self._set_eval_mode()
    #     
    #     with torch.no_grad():
    #         map_tensor = torch.FloatTensor(beatmap_data).to(self.device)
    #         batch_size, seq_len = map_tensor.shape[:2]
    #         
    #         # Initialize with zeros for autoregressive generation
    #         generated = torch.zeros(batch_size, seq_len, 4, device=self.device)
    #         
    #         # Generate sequence autoregressively
    #         for t in range(1, seq_len):
    #             # Use previous predictions as input
    #             pos_dist, keys, _ = self.transformer(map_tensor, generated)
    #             
    #             # Sample from position distribution
    #             mu_xy = pos_dist[:, t-1, :2]
    #             sig_xy = torch.exp(pos_dist[:, t-1, 2:]) * temperature
    #             pos_sample = torch.normal(mu_xy, sig_xy)
    #             
    #             # Sample keypresses
    #             key_probs = torch.sigmoid(keys[:, t-1, :] / temperature)
    #             key_sample = torch.bernoulli(key_probs)
    #             
    #             # Store generated values
    #             generated[:, t, :2] = pos_sample
    #             generated[:, t, 2:] = key_sample
    #     
    #     self._set_train_mode()
    #     return generated.cpu().numpy()
