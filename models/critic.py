import torch
import torch.nn as nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from .jit_lstm import JitLSTM
from .model_utils import TransformerArgs


# TODO write own lstm that doens't rely on cudnn
class ReplayCritic(nn.Module):
    def __init__(self, input_size, lstm_hidden_size=128, lstm_layers=2, dropout=0.1):
        super().__init__()

        self.lstm = JitLSTM(
            input_size=input_size + 2,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=0.0,
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
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
        _, (h_n, _) = self.lstm(x)

        h = h_n[-1]  # (B, lstm_hidden_dim)

        # TODO! don't pool over time why we losing temporal info

        score = self.mlp(h).squeeze(-1)  # (B,)
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
