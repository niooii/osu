from models.vae.vae import OsuReplayVAE

class OsuReplayAVAE(OsuReplayVAE):
    def _train_epoch(self, epoch, total_epochs, **kwargs):
