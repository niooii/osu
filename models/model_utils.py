from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
import torch

from .gan import OsuReplayGAN
from .keys import OsuKeyModel
from .rnn import OsuReplayRNN


@dataclass
class TransformerArgs:
    # Transformer embedding dimension
    embed_dim: int = 128
    # Number of transformer layers  
    transformer_layers: int = 6
    # Feed-forward dimension in transformer layers
    ff_dim: int = 1024
    # Number of attention heads
    attn_heads: int = 8

    def to_dict(self) -> dict:
        """Convert TransformerArgs to dictionary for serialization."""
        return {
            'embed_dim': self.embed_dim,
            'transformer_layers': self.transformer_layers,
            'ff_dim': self.ff_dim,
            'attn_heads': self.attn_heads
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'TransformerArgs':
        """Create TransformerArgs from dictionary."""
        return cls(
            embed_dim=data.get('embed_dim', 128),
            transformer_layers=data.get('transformer_layers', 6),
            ff_dim=data.get('ff_dim', 1024),
            attn_heads=data.get('attn_heads', 8)
        )


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


