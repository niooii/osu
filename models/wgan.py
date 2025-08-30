from os import terminal_size
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tqdm
from critic import ReplayCritic, gradient_penalty
from torch.nn.attention import SDPBackend, sdpa_kernel

import osu.dataset as dataset
from models.base import OsuModel
from models.standard_encoder_t import MapEncoder
from models.vae.decoder_t import ReplayDecoderT
from models.vae.vae import OsuReplayVAE


# same as the vae decoder transformer variant, but the z is just the noise dim
class ReplayGenerator(ReplayDecoderT):
    def __init__(
        self,
        embed_dim: int,
        transformer_layers: int,
        ff_dim: int,
        attn_heads: int,
        past_frames: int,
        future_frames: int,
        noise_dim=32,
    ):
        super().__init__(
            latent_dim=noise_dim,
            embed_dim=embed_dim,
            transformer_layers=transformer_layers,
            ff_dim=ff_dim,
            attn_heads=attn_heads,
            past_frames=past_frames,
            future_frames=future_frames,
        )

    # z: (B, noise_dim)
    def forward(self, map_embeddings, z):
        return super().forward(map_embeddings, z)


# an implementation of a WGAN with a gp critic
class OsuReplayWGAN(OsuModel):
    def __init__(
        self,
        batch_size=64,
        device=None,
        frame_window=(20, 70),
        noise_std=0.0,
        noise_dim=64,
        critic_steps=3,
        lambda_gp=10.0,  # weight for gradient penalty
        lambda_cons=1.0,  # weight for G-C consistency loss
        lambda_pos=1.0,  # weight for position loss via mse
        lr_vae=1e-4,
        lr_g=1e-4,
        lr_c=1e-4,
        betas_vae=(0.9, 0.999),
        betas_gan=(0.5, 0.9),
    ):
        self.noise_dim = noise_dim
        self.critic_steps = critic_steps
        self.lambda_gp = lambda_gp
        self.lambda_cons = lambda_cons
        self.lambda_pos = lambda_pos
        self.lr_vae = lr_vae
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.betas_vae = betas_vae
        self.betas_gan = betas_gan

        super().__init__(
            batch_size=batch_size,
            device=device,
            frame_window=frame_window,
            noise_std=noise_std,
            compile=False,
        )

    def _initialize_models(self, **kwargs):
        # get the encoder & decoder from parent
        super()._initialize_models(**kwargs)

        self.encoder = MapEncoder(
            input_size=len(dataset.INPUT_FEATURES)
            # TODO NO HARDCODE!!
        )

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
            window_feat_dim=window_feat_dim,
        )

    def _initialize_optimizers(self):
        self.opt_G = optim.AdamW(self.generator.parameters(), lr=1e-4)
        self.opt_C = optim.AdamW(self.critic.parameters(), lr=1e-4)

    def _get_state_dict(self):
        sd = super()._get_state_dict()
        sd.update(
            {
                "critic": self.critic.state_dict(),
                "noise_dim": self.noise_dim,
                "critic_steps": self.critic_steps,
                "lambda_gp": self.lambda_gp,
                "generator": self.generator.state_dict(),
                # TODO! save more properties
            }
        )
        return sd

    def _load_state_dict(self, checkpoint):
        # Strip compile prefixes since AVAE uses compile=False but checkpoint is from compiled model
        if "encoder" in checkpoint:
            checkpoint["encoder"] = {
                k.replace("_orig_mod.", ""): v for k, v in checkpoint["encoder"].items()
            }
        if "decoder" in checkpoint:
            checkpoint["decoder"] = {
                k.replace("_orig_mod.", ""): v for k, v in checkpoint["decoder"].items()
            }

        super()._load_state_dict(checkpoint)
        if "generator" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator"])
        else:
            print("No generator weights found, not loading generator")

        if "critic" in checkpoint:
            self.critic.load_state_dict(checkpoint["critic"])
        else:
            print("No critic weights found, not loading critic")

    # in case the VAE (encoder/decoder nets) aren't loaded from pretrained weights, train them using this
    def train_vae_nets(self, **kwargs):
        self.unfreeze_model(model=self.encoder)
        self.unfreeze_model(model=self.decoder)
        super().train(**kwargs)

    # overwrite base vae training function (this assumes the VAE has been pretrained, trains the generator and critic)
    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_adv_loss = 0.0
        epoch_G_loss = 0.0
        epoch_C_loss = 0.0

        device = self.device or torch.device("cpu")

        # freeze encoder and decoder, assuming theyve been pretrained
        self.freeze_model(model=self.encoder)
        self.freeze_model(model=self.decoder)

        for j, (batch_x, batch_y_pos) in enumerate(self.train_loader):
            status_prefix = f"{j}/{len(self.train_loader)} "
            self._set_custom_train_status(status_prefix)

            batch_x = batch_x.to(device)
            batch_y_pos = batch_y_pos.to(device)

            # BATCH SIZE
            B = batch_x.shape[0]

            # (B, T, input_size*window_size)
            windowed = self.create_windowed_features(batch_x)

            critic_loss_accum = 0.0

            with torch.no_grad():
                # (B, L), (B, L), L = latent dim
                mu, logvar = self.encoder(windowed, batch_y_pos)

                # (B, L)
                z = self.reparameterize(mu, logvar)

            # train the critic for n steps per epoch
            for i in range(self.critic_steps):
                self._set_custom_train_status(
                    status_prefix + f"Critic: {i}/{self.critic_steps}"
                )

                # sample xi noise for generator
                # (B, noise_dim)
                xi_c = torch.randn(B, self.noise_dim, device=device)

                # (B, T, 2)
                fake_pos = self.generator(windowed, z, xi_c).detach()

                # (B,)
                real_score = self.critic(windowed, batch_y_pos)
                # (B,)
                fake_score = self.critic(windowed, fake_pos)

                # lambda term is already multiplied in here
                gp = gradient_penalty(
                    self.critic,
                    windowed,
                    real_pos=batch_y_pos,
                    fake_pos=fake_pos,
                    device=device,
                    lambda_gp=10,
                )

                # total wgan-gp critic loss
                loss = -(real_score.mean() - fake_score.mean()) + gp

                self.opt_C.zero_grad()
                loss.backward()
                self.opt_C.step()

                critic_loss_accum += loss.item()

            # train the generator now
            self._set_custom_train_status(status_prefix + f"Generator")

            xi = torch.randn(B, self.noise_dim, device=device)
            fake = self.generator(windowed, z, xi)

            # TODO! add consistency loss prob
            adv_loss = -self.critic(windowed, fake).mean()
            pos_loss = self.lambda_pos * F.smooth_l1_loss(
                fake, batch_y_pos, reduction="mean"
            )

            gen_loss = adv_loss + pos_loss

            self.opt_G.zero_grad()
            gen_loss.backward()
            self.opt_G.step()

            epoch_C_loss += critic_loss_accum / max(1, self.critic_steps)
            epoch_G_loss += gen_loss.item()
            epoch_recon_loss += pos_loss.item()
            epoch_adv_loss += adv_loss.item()

        # anneal KL at epoch end
        # not training the VAE though, maybe itll help idk
        # self.annealer.step()

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

            embeddings = self.encoder(beatmap_data)

            pos = self.generator(embeddings, noise)

        return pos.cpu().numpy()
