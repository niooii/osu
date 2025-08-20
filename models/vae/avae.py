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
    def __init__(self, window_feat_dim, proj_dim=64, lstm_hidden=256, n_layers=2):
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
            critic_steps=5,
            lambda_gp=10.0,
            gamma_enc_adv=0.1,    # weight for encoder adversarial term
            beta_consistency=1.0, # weight for G-D consistency loss
            lr_vae=1e-4,
            lr_g=1e-4,
            lr_c=1e-4,
            betas_vae=(0.9, 0.999),
            betas_gan=(0.5, 0.9)
    ):
        self.xi_dim = xi_dim
        self.critic_steps = critic_steps
        self.lambda_gp = lambda_gp
        self.gamma_enc_adv = gamma_enc_adv
        self.beta_consistency = beta_consistency
        self.lr_vae = lr_vae
        self.lr_g = lr_g
        self.lr_c = lr_c
        self.betas_vae = betas_vae
        self.betas_gan = betas_gan

        super().__init__(annealer=annealer, batch_size=batch_size, device=device,
                         latent_dim=latent_dim, frame_window=frame_window, noise_std=noise_std)

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
        pass
        
    # ---------- save/load ----------
    def _get_state_dict(self):
        sd = super()._get_state_dict()
        sd.update({
            'critic': self.critic.state_dict(),
            'xi_dim': self.xi_dim,
            'critic_steps': self.critic_steps,
            'lambda_gp': self.lambda_gp,
            'gamma_enc_adv': self.gamma_enc_adv,
            'beta_consistency': self.beta_consistency,
        })
        return sd

    def _load_state_dict(self, checkpoint):
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
        return gp 

    # overwrite base vae training function (this assumes the VAE has been pretrained, trains the generator and critic)
    def _train_epoch(self, epoch, total_epochs, **kwargs):
        epoch_total_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_kl_loss = 0.0
        epoch_G_loss = 0.0
        epoch_C_loss = 0.0

        device = self.device or torch.device("cpu")

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
            
            # train the critic for n steps per epoch
            for i in range(self.critic_steps):
                self._set_custom_train_status(status_prefix + f"Critic: {i}/{self.critic_steps}")
                continue
                critic_loss_accum += loss_C.item()
                
            epoch_C_loss += critic_loss_accum / max(1, self.critic_steps)

        # anneal KL at epoch end
        self.annealer.step()

        nb = len(self.train_loader)
        return {
            'total': epoch_total_loss / nb,
            'recon': epoch_recon_loss / nb,
            'kl': epoch_kl_loss / nb,
            'gen': epoch_G_loss / nb,
            'critic': epoch_C_loss / nb
        }

    # inference is the same as the VAE base class, using the decoder
    # TODO! wait no we want to use the generator 
