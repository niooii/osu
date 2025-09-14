from typing import Optional
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm.auto as tqdm

import osu.dataset as dataset

from ..annealer import Annealer
from ..base import OsuModel
from ..loss import spinner_loss
from .decoder import ReplayDecoder
from .encoder import ReplayEncoder


class OsuReplayVAE(OsuModel):
    def __init__(
        self,
        annealer: Annealer = None,
        batch_size=64,
        device=None,
        latent_dim=48,
        frame_window=(20, 70),
        noise_std=0.0,
        lambda_spin=100.0,
        bi_enc: bool = False,
        bi_dec: bool = False,
        compile: bool = True
    ):
        self.latent_dim = latent_dim
        self.past_frames = frame_window[0]
        self.future_frames = frame_window[1]
        self.noise_std = noise_std
        self.lambda_spin = lambda_spin
        self.bi_enc = bi_enc
        self.bi_dec = bi_dec
        self.annealer = annealer or Annealer(
            total_steps=10, range=(0, 0.3), cyclical=True, stay_max_steps=5
        )

        super().__init__(
            batch_size=batch_size,
            device=device,
            compile=compile
            # past_frames=self.past_frames,
            # future_frames=self.future_frames,
            # latent_dim=self.latent_dim,
            # noise_std=noise_std,
            # annealer=self.annealer
        )

    def _initialize_models(self, **kwargs):
        self.encoder = ReplayEncoder(
            input_size=self.input_size,
            latent_dim=self.latent_dim,
            noise_std=self.noise_std,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
            bidirectional=self.bi_enc,
        )
        self.decoder = ReplayDecoder(
            self.input_size,
            latent_dim=self.latent_dim,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
            bidirectional=self.bi_dec,
        )

    def _initialize_optimizers(self):
        params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        self.optimizer = optim.AdamW(
            params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.001
        )

    # like attention but worse..
    def create_windowed_features_legacy(self, beatmap_features):
        batch_size, seq_len, feat_size = beatmap_features.shape

        # Pad the sequence
        padded = F.pad(
            beatmap_features,
            (0, 0, self.past_frames, self.future_frames, 0, 0),
            mode="constant",
            value=dataset.DEFAULT_NORM_FRAME,
        )

        # Create windows
        windows = []
        for i in range(seq_len):
            # Extract window: [past...current...future]
            window = padded[:, i : i + self.past_frames + 1 + self.future_frames, :]
            # Flatten the window
            window_flat = window.reshape(batch_size, -1)
            windows.append(window_flat)

        return torch.stack(windows, dim=1)

    # i forgot to use default frame :sob:
    def create_windowed_features(self, beatmap_features):
        batch_size, seq_len, feat_size = beatmap_features.shape

        default_frame = torch.tensor(
            dataset.DEFAULT_NORM_FRAME,
            dtype=beatmap_features.dtype,
            device=beatmap_features.device,
        )

        #  left padding for past frames at beginning
        # 2 unsqueeze calls -> [1, 1, 9]
        # expand -> [32, past_frames, 9]
        left_pad = (
            default_frame.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.past_frames, feat_size)
        )

        # right padding
        right_pad = (
            default_frame.unsqueeze(0)
            .unsqueeze(0)
            .expand(batch_size, self.future_frames, feat_size)
        )

        padded = torch.cat([left_pad, beatmap_features, right_pad], dim=1)

        windows = []
        for i in range(seq_len):
            # extract window centered at position i ew ew ew wtjh
            window = padded[:, i : i + self.past_frames + 1 + self.future_frames, :]

            flat = window.reshape(batch_size, -1)

            windows.append(flat)

        return torch.stack(windows, dim=1)

    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0
        epoch_recon_loss = 0
        epoch_kl_loss = 0
        epoch_spinner_loss = 0
        epoch_kl_weighted_loss = 0

        beta = self.annealer.current()
        for i, (batch_x, batch_y_pos) in enumerate(
            tqdm.tqdm(
                self.train_loader,
                disable=True,
                position=1,
                desc=f"Epoch {epoch + 1}/{total_epochs} (Beta: {self.annealer.current()})",
            )
        ):
            batch_x = batch_x.to(self.device)             # (B, T, features)
            batch_y_pos = batch_y_pos.to(self.device)     # (B, T, pos)

            self.optimizer.zero_grad()

            # Forward pass
            reconstructed, mu, logvar = self.forward(batch_x, batch_y_pos)

            # Compute loss
            total_loss, recon_loss, kl_loss, spin_loss = self.loss_function(
                reconstructed, batch_y_pos, mu, logvar, batch_x
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
            epoch_spinner_loss += spin_loss.item()
            epoch_kl_weighted_loss += (beta * kl_loss.item())

            mu_m = mu.mean().item(); mu_s = mu.std().item()
            lv_m = logvar.mean().item(); lv_s = logvar.std().item()
            self._set_custom_train_status(
                f"Batch {i}/{len(self.train_loader)} | mu:{mu_m:.4f}/{mu_s:.4f} lv:{lv_m:.4f}/{lv_s:.4f} β:{beta:.3f}"
            )

        # Calculate average losses
        avg_total_loss = epoch_total_loss / len(self.train_loader)
        avg_recon_loss = epoch_recon_loss / len(self.train_loader)
        avg_kl_loss = epoch_kl_loss / len(self.train_loader)
        avg_spinner_loss = epoch_spinner_loss / len(self.train_loader)
        avg_kl_weighted_loss = epoch_kl_weighted_loss / len(self.train_loader)

        # Step the annealer
        self.annealer.step()

        return {
            "total_loss": avg_total_loss,
            "recon_loss": avg_recon_loss,
            "kl_loss": avg_kl_loss,
            "βkl": avg_kl_weighted_loss,
            "spinner_loss": avg_spinner_loss,
        }

    # allow backpropogatino through sampling
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, beatmap_features, positions):
        windowed_features = self.create_windowed_features(beatmap_features)

        mu, logvar = self.encoder(windowed_features, positions)

        z = self.reparameterize(mu, logvar)

        reconstructed = self.decoder(windowed_features, z)

        return reconstructed, mu, logvar

    # recon + kl term + spinner term
    def loss_function(self, reconstructed, original, mu, logvar, beatmap_features):
        # (B, T, 2) vs (B, T, 2) — average over all elements for stable magnitudes
        recon_loss = F.mse_loss(reconstructed, original, reduction="mean")

        # KL per-sample (sum over latent dims), then mean over batch
        # mu/logvar: (B, latent_dim)
        kld_per = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)  # (B)
        kld = kld_per.mean()

        # Calculate spinner loss (scalar), weighted
        spin_loss = spinner_loss(original, reconstructed, beatmap_features) * self.lambda_spin
        
        total_loss = recon_loss + self.annealer(kld) + spin_loss

        return total_loss, recon_loss, kld, spin_loss

    def _get_state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "decoder": self.decoder.state_dict(),
            "latent_dim": self.latent_dim,
            "input_size": self.input_size,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
            "lambda_spin": self.lambda_spin,
            "bi_enc": self.bi_enc,
            "bi_dec": self.bi_dec,
        }

    def _load_state_dict(self, checkpoint):
        self.encoder.load_state_dict(checkpoint["encoder"])
        self.decoder.load_state_dict(checkpoint["decoder"])

    @classmethod
    def load(cls, path: str, device: Optional[torch.device] = None, **kwargs):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load state dict
        checkpoint = torch.load(path, map_location=device)

        # load hyperparams (should make this automatic sometime)
        vae_args = {
            "frame_window": (checkpoint["past_frames"], checkpoint["future_frames"]),
            "latent_dim": checkpoint["latent_dim"],
            "lambda_spin": checkpoint.get("lambda_spin", 0.1),  # default to 0.1 for backwards compatibility
            "bi_enc": checkpoint.get("bi_enc", False),
            "bi_dec": checkpoint.get("bi_dec", False),
        }

        instance = cls(device=device, **kwargs, **vae_args)

        instance._load_state_dict(checkpoint)
        instance._set_eval_mode()

        print(f"{cls.__name__} loaded from {path}")
        return instance

    def generate(self, beatmap_data, num_samples=1):
        self._set_eval_mode()

        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

            # create windowed features for generation
            windowed_features = self.create_windowed_features(beatmap_tensor)

            batch_size = beatmap_tensor.shape[0]

            # sample from prior distribution
            z = torch.randn(batch_size, self.latent_dim, device=self.device)

            pos = self.decoder(windowed_features, z)

        return np.pad(pos.cpu().numpy(), ((0, 0), (0, 0), (0, 2)), mode='constant', constant_values=0)
