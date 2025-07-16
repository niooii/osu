import torch
import torch.nn as nn


class PosModel(nn.Module):
    def __init__(self, input_size, noise_std=0.0):
        super(PosModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 128, num_layers=2, batch_first=True, dropout=0.2)
        self.dense1 = nn.Linear(128, 128)
        self.noise_std = noise_std
        self.dense2 = nn.Linear(128, 64)
        self.position = nn.Linear(64, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        features = nn.functional.relu(self.dense1(lstm_out))

        # gaussian noise (only applied during training)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(features) * self.noise_std
            features = features + noise

        features = nn.functional.relu(self.dense2(features))
        pos = self.position(features)

        return pos


class KeyModel(nn.Module):
    def __init__(self, input_size, noise_std=0.0):
        super(KeyModel, self).__init__()
        self.lstm = nn.LSTM(input_size, 64, num_layers=2, batch_first=True, dropout=0.1)
        self.dense1 = nn.Linear(64, 32)
        self.noise_std = noise_std
        self.keys = nn.Linear(32, 2)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)

        features = nn.functional.relu(self.dense1(lstm_out))

        # gaussian noise (only applied during training)
        # if self.training and self.noise_std > 0:
        #     noise = torch.randn_like(features) * self.noise_std
        #     features = features + noise

        keys = self.keys(features)

        return keys