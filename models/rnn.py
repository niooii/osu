import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import osu.dataset as dataset
from .model import PosModel
from .base import OsuModel


class OsuReplayRNN(OsuModel):
    def __init__(self, batch_size=64, device=None, noise_std=0.0):
        # Store RNN-specific parameters before calling super().__init__
        self.noise_std = noise_std
        self.output_size = len(dataset.OUTPUT_FEATURES)
        
        # Call parent constructor which will call our abstract methods
        super().__init__(batch_size=batch_size, device=device, noise_std=noise_std)
    
    def _initialize_models(self, **kwargs):
        """Initialize RNN position model."""
        self.pos_model = PosModel(self.input_size, self.noise_std)
    
    def _initialize_optimizers(self):
        """Initialize RNN optimizer."""
        self.pos_optimizer = optim.AdamW(self.pos_model.parameters(), lr=0.008, weight_decay=0.001)
        self.pos_criterion = nn.SmoothL1Loss()
    
    def _train_epoch(self, epoch, total_epochs, **kwargs):
        """Train RNN for one epoch with training and validation phases."""
        # Training phase
        self.pos_model.train()
        epoch_train_loss = 0

        for batch_x, batch_y_pos in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} (Train)'):
            batch_x = batch_x.to(self.device)
            batch_y_pos = batch_y_pos.to(self.device)

            # Train position model
            self.pos_optimizer.zero_grad()
            pos = self.pos_model(batch_x)
            pos_loss = self.pos_criterion(pos, batch_y_pos)
            pos_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.pos_model.parameters(), max_norm=1.0)
            self.pos_optimizer.step()

            epoch_train_loss += pos_loss.item()

        # Validation phase
        self.pos_model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y_pos in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y_pos = batch_y_pos.to(self.device)

                pos = self.pos_model(batch_x)
                pos_loss = self.pos_criterion(pos, batch_y_pos)
                epoch_test_loss += pos_loss.item()

        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(self.train_loader)
        avg_test_loss = epoch_test_loss / len(self.test_loader)

        return {
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss
        }

    def _get_state_dict(self):
        """Get state dictionary for saving RNN model."""
        return {
            'pos_model': self.pos_model.state_dict()
        }
    
    def _load_state_dict(self, checkpoint):
        """Load state dictionary for RNN model."""
        self.pos_model.load_state_dict(checkpoint['pos_model'])

    def generate(self, beatmap_data):
        """Generate position data only - returns (x, y) positions"""
        self._set_eval_mode()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            
            # Get position output only
            pos = self.pos_model(beatmap_tensor)

        self._set_train_mode()
        return pos.cpu().numpy()
