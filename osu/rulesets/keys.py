from enum import IntFlag, auto
import numpy as np


class Keys(IntFlag):
    M1 = 1
    M2 = 2,
    K1 = 4,
    K2 = 8,
    SMOKE = 16


def convert_single_key_to_dual(model_output):
    """
    Convert model output from [x, y, key_down] to [x, y, k1, k2]
    For now, key_down maps to k1 and k2 is always 0
    
    Args:
        model_output: numpy array of shape (..., 3) with [x, y, key_down]
        
    Returns:
        numpy array of shape (..., 4) with [x, y, k1, k2]
    """
    # Extract components
    x_y = model_output[..., :2]  # x, y coordinates
    key_down = model_output[..., 2:3]  # key_down signal
    
    # Create k1 (same as key_down) and k2 (zeros)
    k1 = key_down
    k2 = np.zeros_like(key_down)
    
    # Combine into [x, y, k1, k2] format
    return np.concatenate([x_y, k1, k2], axis=-1)
