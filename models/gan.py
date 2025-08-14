import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tqdm

import osu.dataset as dataset
from .model import PosModel, KeyModel
from .base import OsuModel


class Generator(nn.Module):
    def __init__(self, input_size, noise_dim=32):
        super(Generator, self).__init__()

        self.pos_model = PosModel(input_size + noise_dim)
        self.noise_dim = noise_dim

    def forward(self, beatmap_features, noise=None):
        if noise is None:
            batch_size, seq_len, _ = beatmap_features.shape
            noise = torch.randn(batch_size, seq_len, self.noise_dim,
                                device=beatmap_features.device)

        x = torch.concat([beatmap_features, noise], dim=-1)
        pos = self.pos_model(x)

        return pos


class Discriminator(nn.Module):
    def __init__(self, input_size):
        super(Discriminator, self).__init__()

        # beatmap features + cursor positions (x, y only)
        combined_input_size = input_size + 2  # 2 for x, y position

        self.lstm = nn.LSTM(combined_input_size, 64, num_layers=2, batch_first=True, dropout=0.2)

        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 1)

        self.dropout = nn.Dropout(0.1)

    def forward(self, beatmap_features, position_output):
        # Concatenate beatmap features with position output (x, y only)
        x = torch.cat([beatmap_features, position_output], dim=-1)

        # summary of entire sequence
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]

        # Dense layers with dropout
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.classifier(x)

        return x


class OsuReplayGAN(OsuModel):
    def __init__(self, batch_size=64, device=None):
        # Call parent constructor which will call our abstract methods
        super().__init__(batch_size=batch_size, device=device)
    
    def _initialize_models(self, **kwargs):
        """Initialize GAN generator and discriminator models."""
        self.generator = Generator(self.input_size)
        self.discriminator = Discriminator(self.input_size)
        
        # Initialize loss functions
        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pos_criterion = nn.SmoothL1Loss()
    
    def _initialize_optimizers(self):
        """Initialize separate optimizers for generator and discriminator."""
        self.gen_optimizer = optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
        self.disc_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.001)
    
    def _extract_target_data(self, output_data):
        """Extract position data (x, y) from output data for GAN training."""
        return output_data[:, :, :2]  # Take first 2 features (x, y)
    
    def _get_target_data_name(self):
        """Return description of target data type."""
        return "position data"

    def _train_epoch(self, epoch, total_epochs, lambda_pos=0.01, **kwargs):
        """Train GAN for one epoch with alternating generator and discriminator training."""
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_gen_pos_loss = 0

        for batch_x, batch_y_pos in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} (GAN)'):
            batch_x = batch_x.to(self.device)
            batch_y_pos = batch_y_pos.to(self.device)
            batch_size = batch_x.size(0)

            # Real and fake labels
            real_labels = torch.ones(batch_size, 1, device=self.device)
            fake_labels = torch.zeros(batch_size, 1, device=self.device)

            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.disc_optimizer.zero_grad()

            # Real data
            real_output = self.discriminator(batch_x, batch_y_pos)
            real_loss = self.adversarial_loss(real_output, real_labels)

            noise = torch.randn(batch_size, batch_x.shape[1], self.generator.noise_dim).to(self.device)

            # Fake data
            fake_pos = self.generator(batch_x, noise)
            fake_output = self.discriminator(batch_x, fake_pos.detach())
            fake_loss = self.adversarial_loss(fake_output, fake_labels)

            # Total discriminator loss
            disc_loss = (real_loss + fake_loss) / 2
            disc_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
            self.disc_optimizer.step()

            # -----------------
            #  Train Generator
            # -----------------
            self.gen_optimizer.zero_grad()

            # Generate fake position data
            fake_pos = self.generator(batch_x, noise)

            # Adversarial loss (fool discriminator)
            fake_output = self.discriminator(batch_x, fake_pos)
            adv_loss = self.adversarial_loss(fake_output, real_labels)

            # Reconstruction loss
            pos_loss = self.pos_criterion(fake_pos, batch_y_pos)

            # Total generator loss
            gen_loss = adv_loss + lambda_pos * pos_loss
            gen_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.generator.parameters(), max_norm=1.0)
            self.gen_optimizer.step()

            epoch_gen_loss += gen_loss.item()
            epoch_disc_loss += disc_loss.item()
            epoch_gen_pos_loss += pos_loss.item()

        # Calculate average losses
        avg_gen_loss = epoch_gen_loss / len(self.train_loader)
        avg_disc_loss = epoch_disc_loss / len(self.train_loader)
        avg_gen_pos_loss = epoch_gen_pos_loss / len(self.train_loader)

        return {
            'gen_loss': avg_gen_loss,
            'disc_loss': avg_disc_loss,
            'gen_pos_loss': avg_gen_pos_loss
        }

    def _get_state_dict(self):
        """Get state dictionary for saving GAN model."""
        return {
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }
    
    def _load_state_dict(self, checkpoint):
        """Load state dictionary for GAN model."""
        self.generator.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])


    @staticmethod
    def load_generator_only(path, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = len(dataset.INPUT_FEATURES)

        # Create generator
        generator = Generator(input_size)

        # Load generator from checkpoint
        checkpoint = torch.load(path, map_location=device)

        if isinstance(checkpoint, dict) and 'generator' in checkpoint:
            generator.load_state_dict(checkpoint['generator'])
        else:
            generator.load_state_dict(checkpoint)

        generator.to(device)
        generator.eval()
        print(f"Generator loaded from {path} for inference only")
        return generator

    def generate(self, beatmap_data):
        """Generate position data - returns (x, y) positions"""
        self._set_eval_mode()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

            noise = torch.randn(beatmap_tensor.shape[0], beatmap_tensor.shape[1],
                                self.generator.noise_dim).to(self.device)

            pos = self.generator(beatmap_tensor, noise)

        self._set_train_mode()
        return pos.cpu().numpy()

