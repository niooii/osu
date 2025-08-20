import numpy as np

# small
# xs = np.load(f'.datasets/xs_142_07-21_21-50-55.npy')
# ys = np.load(f'.datasets/ys_142_07-21_21-50-55.npy')

# big
xs = np.load(f".datasets/xs_642_08-13_01-26-33.npy")
ys = np.load(f".datasets/ys_642_08-13_01-26-33.npy")

import torch

BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"

from models.vae.vae import OsuReplayVAE

print("Creating model...")

from models.annealer import Annealer

annealer = Annealer(total_steps=10, range=(0, 1.2), cyclical=True, stay_max_steps=5)

vae_args = {
    "batch_size": BATCH_SIZE,
    "annealer": annealer,
    "noise_std": 0.045,
    "frame_window": (20, 70),
    "latent_dim": 48,
}

vae = OsuReplayVAE(**vae_args)

vae.load_data(xs, ys)

EPOCHS = 60

vae.train(epochs=EPOCHS, save_every=3)
vae.save()

vae.plot_losses()

