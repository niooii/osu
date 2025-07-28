import numpy as np

# small
# xs = np.load(f'.datasets/xs_142_07-21_21-50-55.npy')
# ys = np.load(f'.datasets/ys_142_07-21_21-50-55.npy')

# big
xs = np.load(f'.datasets/xs_5000_07-21_23-28-07.npy')
ys = np.load(f'.datasets/ys_5000_07-21_23-28-07.npy')

import torch

BATCH_SIZE = 128

device = 'cuda' if torch.cuda.is_available() else 'cpu'

from models.vae import OsuReplayVAE
print("Creating model...")
vae = OsuReplayVAE(batch_size=BATCH_SIZE)
vae.load_data(xs, ys)
VAE_EPOCHS = 3

# Train the VAE
for i in range(15):
    vae.train(epochs=VAE_EPOCHS)
    vae.save()

vae.plot_losses()