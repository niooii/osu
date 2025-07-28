import numpy as np
import torch
from typing import Union, Optional

from .rnn import OsuReplayRNN
from .gan import OsuReplayGAN
from .keys import OsuKeyModel


# wrapper around the pos and key models generate functions, returns [[x, y, k1, k2]]
def generate_play(
        beatmap_data,
        position_model: Union[OsuReplayRNN, OsuReplayGAN],
        key_model: OsuKeyModel
) -> np.ndarray:
    # gen position data (x, y)
    position_data = position_model.generate(beatmap_data)

    # gen key data (k1, k2)
    key_data = key_model.generate(beatmap_data)

    # concatenate along feature dim (x, y, k1, k2)
    return np.concatenate([position_data, key_data], axis=-1)


def generate_play_rnn(beatmap_data, rnn_model: OsuReplayRNN, key_model: OsuKeyModel) -> np.ndarray:
    """
    Convenience function for generating play data using RNN for positions.
    
    Args:
        beatmap_data: Beatmap time series data
        rnn_model: RNN model for generating cursor positions
        key_model: KeyModel for generating key presses
        
    Returns:
        np.ndarray: Complete replay data in [x, y, k1, k2] format
    """
    return generate_play(beatmap_data, rnn_model, key_model)


def generate_play_gan(beatmap_data, gan_model: OsuReplayGAN, key_model: OsuKeyModel) -> np.ndarray:
    """
    Convenience function for generating play data using GAN for positions.
    
    Args:
        beatmap_data: Beatmap time series data
        gan_model: GAN model for generating cursor positions
        key_model: KeyModel for generating key presses
        
    Returns:
        np.ndarray: Complete replay data in [x, y, k1, k2] format
    """
    return generate_play(beatmap_data, gan_model, key_model)
