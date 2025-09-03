from os import terminal_size
from typing import Optional

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.attention import SDPBackend, sdpa_kernel

import osu.dataset as dataset
from models.annealer import Annealer
from models.base import OsuModel
from models.critic import ReplayCritic, gradient_penalty
from models.vae.decoder import ReplayDecoder

# same as the vae decoder variant, but the z is just the noise dim
class ReplayGenerator(ReplayDecoder):
    def __init__(
        self,
        past_frames: int,
        future_frames: int,
        noise_dim=32,
    ):
        super().__init__(
            input_size=len(dataset.INPUT_FEATURES),
            latent_dim=noise_dim,
            past_frames=past_frames,
            future_frames=future_frames,
        )


# an implementation of a WGAN with a gp critic
class OsuReplayWGAN(OsuModel):
    def __init__(
        self,
        batch_size=64,
        device=None,
        frame_window=(20, 70),
        noise_dim=64,
        critic_steps=3,
        lambda_gp=10.0,
        lambda_pos_annealer=None,
        lr_vae=1e-4,
        lr_g=1e-4,
        lr_c=1e-4,
        betas_gan=(0.5, 0.9),
    ):
        self.past_frames = frame_window[0]
        self.future_frames = frame_window[1]
        self.noise_dim = noise_dim
        self.critic_steps = critic_steps
        self.lambda_gp = lambda_gp
        self.lr_vae = lr_vae
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.betas_gan = betas_gan
        self.lambda_pos_annealer = lambda_pos_annealer or Annealer(
            total_steps=20, range=(10.0, 1.0), cyclical=False
        )

        super().__init__(
            batch_size=batch_size,
            device=device,
            frame_window=frame_window,
            compile=False,
        )

    def _initialize_models(self, **kwargs):
        # generator
        self.generator = ReplayGenerator(
            noise_dim=self.noise_dim,
            past_frames=self.past_frames,
            future_frames=self.future_frames,
        )

        # windowed features per timestep dimension
        window_size = self.past_frames + 1 + self.future_frames
        window_feat_dim = self.input_size * window_size

        # scores a play on realness
        self.critic = ReplayCritic(
            input_size=window_feat_dim,
        )

    def _initialize_optimizers(self):
        self.opt_G = optim.AdamW(
            self.generator.parameters(), lr=self.lr_g, betas=self.betas_gan
        )
        self.opt_C = optim.AdamW(
            self.critic.parameters(), lr=self.lr_c, betas=self.betas_gan
        )

    def _get_state_dict(self):
        return {
            "critic": self.critic.state_dict(),
            "generator": self.generator.state_dict(),
            "noise_dim": self.noise_dim,
            "past_frames": self.past_frames,
            "future_frames": self.future_frames,
        }

    def _load_state_dict(self, checkpoint):
        if "generator" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator"])
        else:
            print("No generator weights found, not loading generator")

        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        else:
            print("No critic weights found, not loading critic")

    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_G_loss = 0.0
        epoch_C_loss = 0.0

        device = self.device or torch.device("cpu")

        for j, (batch_x, batch_y_pos) in enumerate(self.train_loader):
            status_prefix = f"{j}/{len(self.train_loader)} (λ_pos: {self.lambda_pos_annealer.current():.2f}) "
            self._set_custom_train_status(status_prefix)

            batch_x = batch_x.to(device)
            batch_y_pos = batch_y_pos.to(device)

            windowed_features = self.create_windowed_features(batch_x)

            # BATCH SIZE
            B = batch_x.shape[0]

            critic_loss_accum = 0.0

            # train the critic for n steps per epoch
            for i in range(self.critic_steps):
                self._set_custom_train_status(
                    status_prefix + f"Critic: {i}/{self.critic_steps}"
                )

                # sample xi noise for generator
                # (B, noise_dim)
                xi_c = torch.randn(B, self.noise_dim, device=device)

                # (B, T, 2)
                fake_pos = self.generator(windowed_features, xi_c).detach()

                # (B,)
                real_score = self.critic(windowed_features, batch_y_pos)
                # (B,)
                fake_score = self.critic(windowed_features, fake_pos)

                # lambda term is already multiplied in here
                gp = gradient_penalty(
                    self.critic,
                    windowed_features,
                    real_pos=batch_y_pos,
                    fake_pos=fake_pos,
                    device=device,
                    lambda_gp=self.lambda_gp,
                )

                # total wgan-gp critic loss
                loss = -(real_score.mean() - fake_score.mean()) + gp

                self.opt_C.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
                self.opt_C.step()

                critic_loss_accum += loss.item()

            # train the generator now
            self._set_custom_train_status(status_prefix + f"Generator")

            xi = torch.randn(B, self.noise_dim, device=device)
            fake = self.generator(windowed_features, xi)

            # TODO! add consistency loss prob
            # or mse is good enough idk lol
            adv_loss = -self.critic(windowed_features, fake).mean()
            pos_loss = self.lambda_pos_annealer.current() * F.smooth_l1_loss(
                fake, batch_y_pos, reduction="mean"
            )

            gen_loss = adv_loss + pos_loss

            self.opt_G.zero_grad()
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.opt_G.step()

            epoch_C_loss += critic_loss_accum / max(1, self.critic_steps)
            epoch_G_loss += gen_loss.item()
            epoch_recon_loss += pos_loss.item()
            epoch_adv_loss += adv_loss.item()

        self.lambda_pos_annealer.step()

        nb = len(self.train_loader)
        return {
            "recon": epoch_recon_loss / nb,
            "adv": epoch_adv_loss / nb,
            "gen": epoch_G_loss / nb,
            "critic": epoch_C_loss / nb,
        }

    def generate(self, beatmap_data, **kwargs):
        self._set_eval_mode()

        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

            batch_size = beatmap_tensor.shape[0]

            noise = torch.randn(batch_size, self.noise_dim, device=self.device)

            windowed_features = self.create_windowed_features(beatmap_tensor)

            pos = self.generator(windowed_features, noise)

        return pos.cpu().numpy()

    # copied from vae
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
            window = padded[:, i : i + self.past_frames + 1 + self.future_frames, :]

            flat = window.reshape(batch_size, -1)

            windows.append(flat)

        return torch.stack(windows, dim=1)
