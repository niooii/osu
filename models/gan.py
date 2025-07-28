import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import osu.dataset as dataset
from .model import PosModel, KeyModel


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


class OsuReplayGAN:
    def __init__(self, batch_size=64, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.input_size = len(dataset.INPUT_FEATURES)

        # Initialize models
        self.generator = Generator(self.input_size)
        self.discriminator = Discriminator(self.input_size)

        self.generator.to(self.device)
        self.discriminator.to(self.device)

        if hasattr(torch, 'compile'):
            self.generator = torch.compile(self.generator)
            self.discriminator = torch.compile(self.discriminator)

        self.gen_optimizer = optim.AdamW(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999), weight_decay=0.001)
        self.disc_optimizer = optim.AdamW(self.discriminator.parameters(), lr=0.0001, betas=(0.5, 0.999), weight_decay=0.001)

        self.adversarial_loss = nn.BCEWithLogitsLoss()
        self.pos_criterion = nn.SmoothL1Loss()

        # Training history
        self.gen_losses = []
        self.disc_losses = []
        self.gen_pos_losses = []

        # Data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None

        print(f"GAN Models initialized on {self.device}")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)}")

    def load_data(self, input_data, output_data, test_size=0.2):
        """Load data - splits output_data internally to use only position data (x, y)"""
        # Extract only position data (x, y) from output_data
        position_data = output_data[:, :, :2]  # Take first 2 features (x, y)

        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, position_data, test_size=test_size, random_state=42
        )

        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))

        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")

    def train(self, epochs: int, lambda_pos=0.01):
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")

        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            epoch_gen_pos_loss = 0

            for batch_x, batch_y_pos in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{epochs} (GAN)'):
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

            self.gen_losses.append(avg_gen_loss)
            self.disc_losses.append(avg_disc_loss)
            self.gen_pos_losses.append(avg_gen_pos_loss)

            print(f'Epoch {epoch + 1}/{epochs}, Gen: {avg_gen_loss:.4f} (Pos: {avg_gen_pos_loss:.4f}), Disc: {avg_disc_loss:.4f}')

    def plot_losses(self):
        plt.figure(figsize=(12, 8))

        # Total losses
        plt.subplot(2, 2, 1)
        plt.plot(self.gen_losses, label='Generator Total')
        plt.plot(self.disc_losses, label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Total Loss')
        plt.legend()

        # Generator position loss
        plt.subplot(2, 2, 2)
        plt.plot(self.gen_pos_losses, label='Generator Position')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Generator Position Loss')
        plt.legend()

        # Combined view
        plt.subplot(2, 2, 3)
        plt.plot(self.gen_losses, label='Gen Total', linestyle='-')
        plt.plot(self.disc_losses, label='Discriminator', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('GAN Losses')
        plt.legend()

        # Detailed breakdown
        plt.subplot(2, 2, 4)
        plt.plot(self.gen_pos_losses, label='Gen Position', linestyle='--')
        plt.plot(self.disc_losses, label='Discriminator', linestyle='-')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Detailed View')
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save(self, path_prefix=None):
        if path_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path_prefix = f'.trained/gan_{timestamp}'

        if not os.path.exists('.trained'):
            os.makedirs('.trained')

        # Save both models
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, f'{path_prefix}.pt')
        torch.save({
            'generator': self.generator.state_dict(),
            'discriminator': self.discriminator.state_dict()
        }, '.trained/gan_most_recent.pt')
        print(f"GAN models saved to {path_prefix}.pt")

    @staticmethod
    def load(path, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create instance without data
        gan = OsuReplayGAN(device=device)

        # Load both models
        checkpoint = torch.load(path, map_location=device)
        gan.generator.load_state_dict(checkpoint['generator'])
        gan.discriminator.load_state_dict(checkpoint['discriminator'])
        gan.generator.eval()
        gan.discriminator.eval()
        print(f"GAN models loaded from {path}")
        return gan

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
        self.generator.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)

            noise = torch.randn(beatmap_tensor.shape[0], beatmap_tensor.shape[1],
                                self.generator.noise_dim).to(self.device)

            pos = self.generator(beatmap_tensor, noise)

            return pos.cpu().numpy()
        
        self.generator.train()

