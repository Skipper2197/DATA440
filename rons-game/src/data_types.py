from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import NDArray

@dataclass
class HeatmapData:
    win: NDArray[np.float64]
    tie: NDArray[np.float64]
    score_diff: NDArray[np.float64]
    game_len: NDArray[np.float64]
    rounds: NDArray[np.float64]

ScoringMode = Literal["cards", "rounds"]
