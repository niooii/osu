import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

import osu.dataset as dataset


class Generator(nn.Module):
    def __init__(self, input_size, output_size, noise_dim=32):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        if noise_dim > 0:
            # LSTM for processing beatmap features + noise
            self.lstm = nn.LSTM(input_size + noise_dim, 128, num_layers=2, batch_first=True, dropout=0.2)
        else:
            # LSTM for processing beatmap features only (no noise)
            self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        
        # Dense layers for generating cursor positions
        self.dense1 = nn.Linear(128, 64)
        self.dense2 = nn.Linear(64, 32)
        self.position = nn.Linear(32, output_size)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, beatmap_features, noise=None):
        if self.noise_dim > 0:
            batch_size, seq_len, _ = beatmap_features.shape
            
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn(batch_size, seq_len, self.noise_dim, device=beatmap_features.device)
            
            # Concatenate beatmap features with noise
            x = torch.cat([beatmap_features, noise], dim=-1)
        else:
            # No noise - use beatmap features directly
            x = beatmap_features
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Dense layers with dropout
        x = F.relu(self.dense1(lstm_out))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = self.position(x)
        
        return x


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size):
        super(Discriminator, self).__init__()
        
        # Input: beatmap features + cursor positions
        combined_input_size = input_size + output_size
        
        # LSTM for processing sequences
        self.lstm = nn.LSTM(combined_input_size, 64, num_layers=2, batch_first=True, dropout=0.2)
        
        # Dense layers for classification
        self.dense1 = nn.Linear(64, 32)
        self.dense2 = nn.Linear(32, 16)
        self.classifier = nn.Linear(16, 1)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, beatmap_features, cursor_positions):
        # Concatenate beatmap features with cursor positions
        x = torch.cat([beatmap_features, cursor_positions], dim=-1)
        
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        
        # Take the last output for classification
        x = lstm_out[:, -1, :]  # Use last timestep
        
        # Dense layers with dropout
        x = F.relu(self.dense1(x))
        x = self.dropout(x)
        x = F.relu(self.dense2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.classifier(x))
        
        return x


class OsuReplayGAN:
    def __init__(self, batch_size=64, device=None, noise_dim=0): 
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.input_size = len(dataset.INPUT_FEATURES)
        self.output_size = len(dataset.OUTPUT_FEATURES)
        
        # Initialize models
        self.generator = Generator(self.input_size, self.output_size, noise_dim)
        self.discriminator = Discriminator(self.input_size, self.output_size)
        
        self.generator.to(self.device)
        self.discriminator.to(self.device)
        
        # Initialize optimizers and losses
        self.gen_optimizer = optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        
        self.adversarial_loss = nn.BCELoss()
        self.reconstruction_loss = nn.MSELoss()
        
        # Training history
        self.gen_losses = []
        self.disc_losses = []
        
        # Data loaders (will be set by load_data)
        self.train_loader = None
        self.test_loader = None
        
        print(f"GAN Models initialized on {self.device} (noise_dim={noise_dim})")
        print(f"Generator parameters: {sum(p.numel() for p in self.generator.parameters() if p.requires_grad)}")
        print(f"Discriminator parameters: {sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad)}")
    
    def load_data(self, input_data, output_data, test_size=0.2):
        # Handle input/output data inconsistency by taking minimum
        min_samples = min(input_data.shape[0], output_data.shape[0])
        input_data = input_data[:min_samples]
        output_data = output_data[:min_samples]
        
        # Train/test split
        x_train, x_test, y_train, y_test = train_test_split(
            input_data, output_data, test_size=test_size, random_state=42
        )
        
        # Create DataLoaders
        train_dataset = TensorDataset(torch.FloatTensor(x_train), torch.FloatTensor(y_train))
        test_dataset = TensorDataset(torch.FloatTensor(x_test), torch.FloatTensor(y_test))
        
        self.train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        
        print(f"Data loaded: {len(train_dataset)} training samples, {len(test_dataset)} test samples")
    
    def train(self, epochs: int, lambda_recon=10.0):
        if self.train_loader is None:
            raise ValueError("No data loaded. Call load_data() first.")
            
        for epoch in range(epochs):
            epoch_gen_loss = 0
            epoch_disc_loss = 0
            
            for batch_x, batch_y in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{epochs} (GAN)'):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                batch_size = batch_x.size(0)
                
                # Real and fake labels
                real_labels = torch.ones(batch_size, 1, device=self.device)
                fake_labels = torch.zeros(batch_size, 1, device=self.device)
                
                # ---------------------
                #  Train Discriminator
                # ---------------------
                self.disc_optimizer.zero_grad()
                
                # Real data
                real_output = self.discriminator(batch_x, batch_y)
                real_loss = self.adversarial_loss(real_output, real_labels)
                
                # Fake data
                fake_positions = self.generator(batch_x)
                fake_output = self.discriminator(batch_x, fake_positions.detach())
                fake_loss = self.adversarial_loss(fake_output, fake_labels)
                
                # Total discriminator loss
                disc_loss = (real_loss + fake_loss) / 2
                disc_loss.backward()
                self.disc_optimizer.step()
                
                # -----------------
                #  Train Generator
                # -----------------
                self.gen_optimizer.zero_grad()
                
                # Generate fake positions
                fake_positions = self.generator(batch_x)
                
                # Adversarial loss (fool discriminator)
                fake_output = self.discriminator(batch_x, fake_positions)
                adv_loss = self.adversarial_loss(fake_output, real_labels)
                
                # Reconstruction loss (maintain beatmap correspondence)
                recon_loss = self.reconstruction_loss(fake_positions, batch_y)
                
                # Total generator loss
                gen_loss = adv_loss + lambda_recon * recon_loss
                gen_loss.backward()
                self.gen_optimizer.step()
                
                epoch_gen_loss += gen_loss.item()
                epoch_disc_loss += disc_loss.item()
            
            # Calculate average losses
            avg_gen_loss = epoch_gen_loss / len(self.train_loader)
            avg_disc_loss = epoch_disc_loss / len(self.train_loader)
            
            self.gen_losses.append(avg_gen_loss)
            self.disc_losses.append(avg_disc_loss)
            
            print(f'Epoch {epoch+1}/{epochs}, Gen Loss: {avg_gen_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}')
    
    def plot_losses(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(self.gen_losses, label='Generator')
        ax1.plot(self.disc_losses, label='Discriminator')
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.set_title("GAN Training Losses")
        ax1.legend()
        
        # Plot them separately for better visibility
        ax2.plot(self.gen_losses, label='Generator', color='blue')
        ax2_twin = ax2.twinx()
        ax2_twin.plot(self.disc_losses, label='Discriminator', color='red')
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Generator Loss", color='blue')
        ax2_twin.set_ylabel("Discriminator Loss", color='red')
        ax2.set_title("GAN Losses (Separate Scales)")
        
        plt.tight_layout()
        plt.show()
    
    def save(self, path_prefix=None):
        if path_prefix is None:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            path_prefix = f'.trained/gan_{timestamp}'
        
        if not os.path.exists('.trained'):
            os.makedirs('.trained')
            
        torch.save(self.generator.state_dict(), f'{path_prefix}_generator.pt')
        torch.save(self.discriminator.state_dict(), f'{path_prefix}_discriminator.pt')
        torch.save(self.generator.state_dict(), '.trained/gan_generator_most_recent.pt')
        torch.save(self.discriminator.state_dict(), '.trained/gan_discriminator_most_recent.pt')
        print(f"GAN models saved with prefix {path_prefix}")
    
    @staticmethod
    def load(generator_path, discriminator_path, noise_dim=0, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Create instance without data
        gan = OsuReplayGAN(device=device, noise_dim=noise_dim)
        gan.generator.load_state_dict(torch.load(generator_path, map_location=device))
        gan.discriminator.load_state_dict(torch.load(discriminator_path, map_location=device))
        gan.generator.eval()
        gan.discriminator.eval()
        print(f"GAN models loaded from {generator_path} and {discriminator_path}")
        return gan
    
    @staticmethod
    def load_generator_only(generator_path, noise_dim=0, device=None):
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_size = len(dataset.INPUT_FEATURES)
        output_size = len(dataset.OUTPUT_FEATURES)
        generator = Generator(input_size, output_size, noise_dim)
        generator.load_state_dict(torch.load(generator_path, map_location=device))
        generator.to(device)
        generator.eval()
        print(f"Generator loaded from {generator_path} for inference only")
        return generator
    
    def generate(self, beatmap_data):
        self.generator.eval()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            return self.generator(beatmap_tensor).cpu().numpy()