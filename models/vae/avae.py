from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

from models.vae.vae import OsuReplayVAE
from models.vae.decoder import ReplayDecoder
import osu.dataset as dataset
from ..annealer import Annealer
from torch.cuda.amp import autocast, GradScaler

# as per the avae paper (https://arxiv.org/pdf/2012.11551), there are seperate generator/decoder models. 
# however the arch is pretty much the same, except there are additional noise dimensions alongside the original latent dims
class ReplayGenerator(ReplayDecoder):
    def __init__(self, input_size, latent_dim=48, xi_dim=16, **kwargs):
        super().__init__(input_size=input_size, latent_dim=latent_dim + xi_dim, **kwargs)

    # z: (B, latent_dim), xi: (B, xi_dim)
    def forward(self, beatmap_features, z, xi):
        z_cat = torch.cat([z, xi], dim=-1)    # (B, latent+xi)
        return super().forward(beatmap_features, z_cat)


class ReplayCritic(nn.Module):
    # avoids BatchNorm and uses LayerNorm and pooling over time.
    def __init__(self, window_feat_dim, proj_dim=64, lstm_hidden=128, n_layers=2):
        super().__init__()
        self.proj = nn.Linear(window_feat_dim, proj_dim)  # compress large windowed features
        self.lstm = nn.LSTM(input_size=proj_dim + 2, hidden_size=lstm_hidden,
                            num_layers=n_layers, batch_first=True, dropout=0.2)
        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden, lstm_hidden // 2),
            nn.LayerNorm(lstm_hidden // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(lstm_hidden // 2, 1)
        )

    def forward(self, windowed, positions):
        proj = self.proj(windowed)                 # (B, T, P)
        x = torch.cat([proj, positions], dim=-1)   # (B, T, P+2)
        hseq, _ = self.lstm(x)                     # (B, T, H)
        h = hseq.mean(dim=1)                       # (B, H)
        score = self.mlp(h).squeeze(-1)            # (B,)
        return score


class OsuReplayAVAE(OsuReplayVAE):
    def __init__(
            self,
            annealer: Annealer = None,
            batch_size=64,
            device=None,
            latent_dim=48,
            frame_window=(20, 70),
            noise_std=0.0,
            xi_dim=16,
            critic_steps=3,
            lambda_gp=10.0,  # weight for gradient penalty
            lambda_cons=1.0, # weight for G-C consistency loss
            lambda_pos=1.0,  # weight for position loss via mse
            lr_vae=1e-4,
            lr_g=1e-4,
            lr_c=1e-4,
            betas_vae=(0.9, 0.999),
            betas_gan=(0.5, 0.9)
    ):
        self.xi_dim = xi_dim
        self.critic_steps = critic_steps
        self.lambda_gp = lambda_gp
        self.lambda_cons = lambda_cons
        self.lambda_pos = lambda_pos
        self.lr_vae = lr_vae
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.betas_vae = betas_vae
        self.betas_gan = betas_gan

        super().__init__(annealer=annealer, batch_size=batch_size, device=device,
                         latent_dim=latent_dim, frame_window=frame_window, noise_std=noise_std, compile=False)
        self.scaler_C = GradScaler()
        self.scaler_G = GradScaler()

    def _initialize_models(self, **kwargs):
        # get the encoder & decoder from parent
        super()._initialize_models(**kwargs)

        # generator
        self.generator = ReplayGenerator(
            input_size=self.input_size,
            latent_dim=self.latent_dim,
            xi_dim=self.xi_dim,
            past_frames=self.past_frames,
            future_frames=self.future_frames
        )

        # windowed features per timestep dimension
        window_size = self.past_frames + 1 + self.future_frames
        window_feat_dim = self.input_size * window_size

        # scores a play on realness
        self.critic = ReplayCritic(
            window_feat_dim=window_feat_dim,
            proj_dim=64,
            lstm_hidden=256,
            n_layers=2
        )

    def _initialize_optimizers(self):
        self.opt_G = optim.AdamW(self.generator.parameters(), lr=1e-4)
        self.opt_C = optim.AdamW(self.critic.parameters(), lr=1e-4)
        
    def _get_state_dict(self):
        sd = super()._get_state_dict()
        sd.update({
            'critic': self.critic.state_dict(),
            'xi_dim': self.xi_dim,
            'critic_steps': self.critic_steps,
            'lambda_gp': self.lambda_gp,
            # TODO! save more properties
        })
        return sd

    def _load_state_dict(self, checkpoint):
        # Strip compile prefixes since AVAE uses compile=False but checkpoint is from compiled model
        if 'encoder' in checkpoint:
            checkpoint['encoder'] = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['encoder'].items()}
        if 'decoder' in checkpoint:
            checkpoint['decoder'] = {k.replace('_orig_mod.', ''): v for k, v in checkpoint['decoder'].items()}
            
        super()._load_state_dict(checkpoint)
        if 'generator' in checkpoint:
            self.generator.load_state_dict(checkpoint['generator'])
        else:
            print("No generator weights found, not loading generator")
            
        if 'critic' in checkpoint:
            self.critic.load_state_dict(checkpoint['critic'])
        else:
            print("No critic weights found, not loading critic")
            
    # in case the VAE (encoder/decoder nets) aren't loaded from pretrained weights, train them using this
    def train_vae_nets(self, **kwargs):
        self.unfreeze_model(model=self.encoder)
        self.unfreeze_model(model=self.decoder)
        super().train(**kwargs)
        
    # calculates the gradient penalty loss - we encourage the model to stay 1-lipschitz so it can better estimate the
    # wasserstein dist
    def gradient_penalty(self, windowed, real_pos, fake_pos, device, lambda_gp=10.0):
        B = real_pos.shape[0]
        # alpha (B,1,1) will broadcast across (T,2)
        alpha = torch.rand(B, 1, 1, device=device)

        # interpolation in position space
        # (B, T, 2)
        x = (alpha * real_pos + (1.0 - alpha) * fake_pos).requires_grad_(True)

        # (B,)
        with torch.backends.cudnn.flags(enabled=False):
            C_x = self.critic(windowed, x)

        # compute gradients of outputs wrt interpolated positions
        # (this works since the gradient distributes)
        # (B, T, 2)
        grads = torch.autograd.grad(
            outputs=C_x.sum(),
            inputs=x,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]

        # flatten per-sample gradients and compute L2 norm per sample
        # (B,)
        grad_norm = grads.reshape(B, -1).norm(2, dim=1)

        gp = ((grad_norm - 1.0) ** 2).mean() * lambda_gp

        del grads, grad_norm, C_x, x
        torch.cuda.empty_cache()

        return gp 

    # overwrite base vae training function (this assumes the VAE has been pretrained, trains the generator and critic)
    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_G_loss = 0.0
        epoch_C_loss = 0.0

        device = self.device or torch.device("cpu")

        # freeze encoder and decoder, assuming theyve been pretrained
        self.freeze_model(model=self.encoder)
        self.freeze_model(model=self.decoder)

        for i, (batch_x, batch_y_pos) in enumerate(self.train_loader): 
            status_prefix = f"{i}/{len(self.train_loader)} "
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
                self._set_custom_train_status(status_prefix + f"Critic: {i}/{self.critic_steps}")
                with autocast():
                    # sample xi noise for generator
                    # (B, xi_dim)
                    xi_c = torch.randn(B, self.xi_dim, device=device)

                    # (B, T, 2)
                    fake_pos = self.generator(windowed, z, xi_c).detach()

                    # (B,)
                    real_score = self.critic(windowed, batch_y_pos)
                    # (B,)
                    fake_score = self.critic(windowed, fake_pos)

                    # lambda term is already multiplied in here
                    gp = self.gradient_penalty(windowed, real_pos=batch_y_pos, fake_pos=fake_pos, device=device, lambda_gp=10) 

                    # total wgan-gp critic loss
                    loss = -(real_score.mean() - fake_score.mean()) + gp
                
                self.opt_C.zero_grad()
                self.scaler_C.scale(loss).backward()  # Scale the loss for fp16
                self.scaler_C.step(self.opt_C)
                self.scaler_C.update()
                torch.cuda.empty_cache()

                critic_loss_accum += loss.item()

            # train the generator now
            self._set_custom_train_status(status_prefix + f"Generator")
            with autocast():
                xi = torch.randn(B, self.xi_dim, device=device)
                fake = self.generator(windowed, z, xi)                # no detach -> grads to generator

                # TODO! add consistency loss prob
                adv_loss = -self.critic(windowed, fake).mean()
                pos_loss = F.smooth_l1_loss(fake, batch_y_pos, reduction='mean')

                gen_loss = adv_loss + self.lambda_pos * pos_loss
            
            self.opt_G.zero_grad()
            self.scaler_G.scale(gen_loss).backward()
            self.scaler_G.step(self.opt_G)
            self.scaler_G.update()

                
            epoch_C_loss += critic_loss_accum / max(1, self.critic_steps)

        # anneal KL at epoch end
        # not training the VAE though, maybe itll help idk
        # self.annealer.step()

        nb = len(self.train_loader)
        return {
            'total': epoch_total_loss / nb,
            'recon': epoch_recon_loss / nb,
            # 'kl': epoch_kl_loss / nb,
            'gen': epoch_G_loss / nb,
            'critic': epoch_C_loss / nb
        }

    # inference is the same as the VAE base class, using the decoder
    # TODO! wait no we want to use the generator 
