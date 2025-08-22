import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

import osu.dataset as dataset
from .model import KeyModel
from .base import OsuModel


class OsuKeyModel(OsuModel):
    def __init__(self, batch_size=64, device=None, noise_std=0.0):
        self.noise_std = noise_std
        
        super().__init__(batch_size=batch_size, device=device, noise_std=noise_std)
    
    def _initialize_models(self, **kwargs):
        self.key_model = KeyModel(self.input_size, self.noise_std)
    
    def _initialize_optimizers(self):
        self.key_optimizer = optim.AdamW(self.key_model.parameters(), lr=0.01, weight_decay=0.001)
        self.key_criterion = nn.BCEWithLogitsLoss()
   
    # overload for keypress features
    def _extract_target_data(self, output_data):
        return output_data[:, :, 2:]  # take last 2 features (k1, k2)
    
    def _train_epoch(self, epoch, total_epochs, **kwargs):
        # Training phase
        self.key_model.train()
        epoch_train_loss = 0

        for batch_x, batch_y in tqdm.tqdm(self.train_loader, desc=f'Epoch {epoch + 1}/{total_epochs} (Keys)'):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self._process_keys(batch_y)

            self.key_optimizer.zero_grad()
            keys = self.key_model(batch_x)
            keys_loss = self.key_criterion(keys, batch_y)
            keys_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.key_model.parameters(), max_norm=1.0)
            self.key_optimizer.step()

            epoch_train_loss += keys_loss.item()

        # Validation phase
        self.key_model.eval()
        epoch_test_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in self.test_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                self._process_keys(batch_y)

                keys = self.key_model(batch_x)
                keys_loss = self.key_criterion(keys, batch_y)
                epoch_test_loss += keys_loss.item()

        # Calculate average losses
        avg_train_loss = epoch_train_loss / len(self.train_loader)
        avg_test_loss = epoch_test_loss / len(self.test_loader)

        return {
            'train_loss': avg_train_loss,
            'test_loss': avg_test_loss
        }

    # we find the dominant key and then swap it to the first position
    # not really needed if training off of one person's data though
    def _process_keys(self, keys_tensor: torch.FloatTensor):
        prev_states = torch.cat([
            torch.zeros(keys_tensor.shape[0], 1, 2, device=keys_tensor.device),
            keys_tensor[:, :-1, :]
        ], dim=1)

        # [batch_size, seq_len, 2]
        presses = torch.logical_and(keys_tensor == 1, prev_states == 0)
        # [batch_size, 2]
        press_counts = presses.sum(dim=1)
        # [batch_size] bool tensor (swap values?)
        swap_mask = press_counts[:, 1] > press_counts[:, 0]  # k2 > k1

        keys_tensor[swap_mask] = torch.flip(keys_tensor[swap_mask], dims=[2])

    def _get_state_dict(self):
        return {
            'key_model': self.key_model.state_dict()
        }
    
    def _load_state_dict(self, checkpoint):
        self.key_model.load_state_dict(checkpoint['key_model'])

    def generate(self, beatmap_data):
        self._set_eval_mode()
        with torch.no_grad():
            beatmap_tensor = torch.FloatTensor(beatmap_data).to(self.device)
            
            # Get key logits
            keys = self.key_model(beatmap_tensor)

            # Apply sigmoid and threshold to get boolean keys
            key_probs = torch.sigmoid(keys)
            keys_bool = (key_probs > 0.5).float()

        self._set_train_mode()
        return keys_bool.cpu().numpy()
