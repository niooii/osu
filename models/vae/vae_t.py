from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm

import osu.dataset as dataset

from ..annealer import Annealer
from ..base import OsuModel
from .decoder_t import ReplayDecoderT
from .encoder_t import ReplayEncoderT


# switch from the RNN based VAEee to a transformer based one
# TODO! save hyperparam dict
class OsuReplayTVAE(OsuModel):
    def __init__(
        self,
        annealer: Annealer = None,
        batch_size=64,
        device=None,
        latent_dim=64,
        embed_dim=128,
        transformer_layers=6,
        ff_dim=1024,
        attn_heads=8,
        noise_std=0.0, 
        frame_window=(40, 90),
        compile: bool = True
    ):
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.transformer_layers = transformer_layers
        self.ff_dim = ff_dim
        self.attn_heads = attn_heads
        self.past_frames = frame_window[0]
        self.future_frames = frame_window[1]
        self.noise_std = noise_std
        self.annealer = annealer or Annealer(
            total_steps=10, range=(0, 0.3), cyclical=True, stay_max_steps=5
        )

        super().__init__(
            batch_size=batch_size,
            device=device,
            compile=compile
        )

    def _initialize_models(self, **kwargs):
        self.encoder = ReplayEncoderT(
            input_size=self.input_size,
            embed_dim=self.embed_dim,
            latent_dim=self.latent_dim,
            transformer_layers=self.transformer_layers,
            ff_dim=self.ff_dim,
            attn_heads=self.attn_heads,
            noise_std=self.noise_std,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
        )
        self.decoder = ReplayDecoderT(
            embed_dim=self.embed_dim,
            latent_dim=self.latent_dim,
            transformer_layers=self.transformer_layers,
            ff_dim=self.ff_dim,
            attn_heads=self.attn_heads,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
        )

    def _initialize_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(
            params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.001
        )


    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0

        for i, (batch_x, batch_y_pos) in enumerate(
            tqdm.tqdm(
                self.train_loader,
                disable=True,
                position=1,
                desc=f"Epoch {epoch + 1}/{total_epochs} (Beta: {self.annealer.current()})",
            )
        ):
            self._set_custom_train_status(f"Batch {i}/{len(self.train_loader)}")
            batch_x = batch_x.to(self.device)             # (B, T, features)
            batch_y_pos = batch_y_pos.to(self.device)     # (B, T, pos)

            self.optimizer.zero_grad()

            # Forward pass
            reconstructed, mu, logvar = self.forward(batch_x, batch_y_pos)

            # Compute loss
            total_loss, recon_loss, kl_loss = self.loss_function(
                reconstructed, batch_y_pos, mu, logvar
            )

            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self.encoder.parameters()) + list(self.decoder.parameters()),
                max_norm=1.0,
            )
            self.optimizer.step()

            epoch_total_loss += total_loss.item()
            epoch_recon_loss += recon_loss.item()
            epoch_kl_loss += kl_loss.item()

        # Calculate average losses
        avg_total_loss = epoch_total_loss / len(self.train_loader)
        avg_recon_loss = epoch_recon_loss / len(self.train_loader)
        avg_kl_loss = epoch_kl_loss / len(self.train_loader)

        # Step the annealer
        self.annealer.step()

        return {
            "total_loss": avg_total_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
        }

    # allow backpropogatino through sampling
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, beatmap_features, positions):
        embeddings, mu, logvar = self.encoder(beatmap_features, positions)

        # Sample latent code
        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(beatmap_features, z)

        return reconstructed, mu, logvar

    # recon + kl term
    def loss_function(self, reconstructed, original, mu, logvar):
        # TODO! am i supposed to avg this? probably?
        recon_loss = F.mse_loss(reconstructed, original, reduction="sum")

        kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        kld /= original.shape[0]

        total_loss = recon_loss + self.annealer(kld)

        return total_loss, recon_loss, kld

    def _get_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "latent_dim": self.latent_dim,
            "embed_dim": self.embed_dim,
            "transformer_layers": self.transformer_layers,
            "ff_dim": self.ff_dim,
            "attn_heads": self.attn_heads,
            "noise_std": self.noise_std,
            "input_size": self.input_size,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
        }

    def _load_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(path, map_location=device)
        
        vae_args = {
            "latent_dim": checkpoint.get("latent_dim", 64),
            "embed_dim": checkpoint.get("embed_dim", 128),
            "transformer_layers": checkpoint.get("transformer_layers", 6),
            "ff_dim": checkpoint.get("ff_dim", 1024),
            "attn_heads": checkpoint.get("attn_heads", 8),
            "noise_std": checkpoint.get("noise_std", 0.0),
            "frame_window": (checkpoint.get("past_frames", 40), checkpoint.get("future_frames", 90)),
        }

        instance = cls(device=device, **kwargs, **vae_args)

        instance._load_state_dict(checkpoint)
        instance._set_eval_mode()

        print(f"{cls.__name__} loaded from {path}")
        return instance

    def generate(self, beatmap_data, num_samples=1, apply_bias_correction=True):
        """Generate position data - returns (x, y) positions"""
        self._set_eval_mode()

        # with torch.no_grad():
        #     beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

        #     batch_size = beatmap_tensor.shape[0]

        #     # sample from prior distribution
        #     z = torch.randn(batch_size, self.latent_dim, device=self.device)

        #     pos = self.decoder(beatmap_tensor, z)

        # self._set_train_mode()

        # return pos.cpu().numpy()
