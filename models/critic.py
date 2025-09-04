import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .model_utils import TransformerArgs


# TODO write own lstm that doens't rely on cudnn
class ReplayCritic(nn.Module):
    def __init__(self, input_size, lstm_hidden_size=128, lstm_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size + 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.mlp = nn.Sequential(
            nn.Linear(lstm_hidden_size, lstm_hidden_size // 2),
            nn.LayerNorm(lstm_hidden_size // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(lstm_hidden_size // 2, 1),
        )

    # TODO try with keys also? prob not gonna end well
    def forward(self, map_features, positions):
        # map_features: (B, T, feat_dim), positions: (B, T, 2),
        # T is chunk length
        x = torch.cat([map_features, positions], dim=-1)  # (B, T, feat_dim+2)
        # _, (h_n, _) = self.lstm(x)
        # h = h_n[-1]  # (B, lstm_hidden_dim)
        # TODO! don't pool over time why we losing temporal info

        # score = self.mlp(h).squeeze(-1)  # (B,)

        h_out, _ = self.lstm(x)      # (B, T, H)
        h = h_out.mean(dim=1)
        score = self.mlp(h).squeeze(-1)
        return score


class ReplayCriticT(nn.Module):
    def __init__(self, input_size, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()

        self.input_proj = nn.Linear(input_size + 2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # makes input (B,T,D)

        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over time after transpose

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(d_model // 2, 1)
        )

    # TODO try with keys also? prob not gonna end well
    def forward(self, windowed, positions):
        # windowed: (B, T, W), positions: (B, T, 2)
        x = torch.cat([windowed, positions], dim=-1)  # (B, T, W+2)
        x = self.input_proj(x)                        # (B, T, d_model)

        x = self.transformer(x)                       # (B, T, d_model)

        # pool over time
        x = x.transpose(1, 2)                         # (B, d_model, T)
        x = self.pool(x).squeeze(-1)                  # (B, d_model)
        score = self.mlp(x).squeeze(-1)               # (B,)

        return score

def gradient_penalty(critic, windowed, real_pos, fake_pos, device, lambda_gp=10.0):
    B = real_pos.shape[0]
    # alpha (B,1,1) will broadcast across (T,2)
    alpha = torch.rand(B, 1, 1, device=device)

    # interpolation in position space
    # (B, T, 2)
    x = (alpha * real_pos + (1.0 - alpha) * fake_pos).requires_grad_(True)

    # (B,)
    with sdpa_kernel(backends=[SDPBackend.MATH]):
        C_x = critic(windowed, x)

    # compute gradients of outputs wrt interpolated positions
    # (this works since the gradient distributes)
    # (B, T, 2)
    grads = torch.autograd.grad(
        outputs=C_x.sum(),
        inputs=x,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # flatten per-sample gradients and compute L2 norm per sample
    # (B,)
    grad_norm = grads.reshape(B, -1).norm(2, dim=1)

    gp = ((grad_norm - 1.0) ** 2).mean() * lambda_gp

    del grads, grad_norm, C_x, x

    return gp


def fd_gradient_penalty(critic, map_features, real_pos, eps=1e-2, n_dirs=1, lambda_gp=10.0):
    B = real_pos.shape[0]
    device = real_pos.device

    # evaluate D(real) once
    D_real = critic(map_features, real_pos)            # (B,)

    penalties = []
    for _ in range(n_dirs):
        # sample random unit direction per-sample
        u = torch.randn_like(real_pos, device=device)  # (B,T,2)
        u_flat = u.view(B, -1)
        u_norm = u_flat.norm(dim=1).view(B, 1, 1).clamp(min=1e-12)
        u = u / u_norm

        x_eps = real_pos + eps * u                      # (B,T,2)
        D_eps = critic(map_features, x_eps)             # (B,)

        # directional derivative approx: (D(x+eps*u) - D(x)) / eps
        dir_deriv = (D_eps - D_real) / eps               # (B,)
        # approximate gradient norm along that random direction
        # encourage | directional_deriv | ≈ 1
        penalties.append(((dir_deriv.abs() - 1.0) ** 2).mean())

    gp = torch.stack(penalties).mean() * lambda_gp
    return gp
