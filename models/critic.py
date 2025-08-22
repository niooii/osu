import torch
import torch.nn as nn
from torch.nn.attention import sdpa_kernel, SDPBackend


# follows the WGAN-GP paper, but this needs to use attention instead of an RNN since CuDNN does not support 
# second order gradients
class ReplayCritic(nn.Module):
    def __init__(self, window_feat_dim, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(window_feat_dim + 2, d_model)
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
        only_inputs=True
    )[0]

    # flatten per-sample gradients and compute L2 norm per sample
    # (B,)
    grad_norm = grads.reshape(B, -1).norm(2, dim=1)

    gp = ((grad_norm - 1.0) ** 2).mean() * lambda_gp

    del grads, grad_norm, C_x, x

    return gp
