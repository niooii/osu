from models.vae.vae import OsuReplayVAE

class OsuReplayAVAE(OsuReplayVAE):
    def loss_function(self, reconstructed, original, mu, logvar):


    def _train_epoch(self, epoch, total_epochs, **kwargs):
