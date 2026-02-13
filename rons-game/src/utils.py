import os
import numpy as np
from numpy.typing import NDArray

def heatmap_file(trials: int, scoring: str) -> str:
    return f'data/heatmaps_{scoring}_{trials}.npz'

def lower_triangle_mask(arr: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.triu(np.ones_like(arr, dtype=bool))

def data_exists(trials: int, scoring: str) -> bool:
    return os.path.exists(heatmap_file(trials, scoring))