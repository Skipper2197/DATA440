import os
import numpy as np
from numpy.typing import NDArray

def lower_triangle_mask(arr: NDArray[np.float64]) -> NDArray[np.bool_]:
    '''
    Return an upper trianlge mask
    '''
    return np.triu(np.ones_like(arr, dtype=bool))  # mask upper triangle to hide lower

def ensure_project_dirs() -> None:
    '''
    Create necessary directories if they don't exist
    '''
    dirs = [
        'data',
        'data/decks',
        'data/results',
        'figures',
        'figures/cards',
        'figures/rounds',
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)

    return
