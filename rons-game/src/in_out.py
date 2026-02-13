import numpy as np
from src.data_types import HeatmapData
from src.utils import heatmap_file

def save_data(data: HeatmapData, trials: int, scoring: str) -> None:
    np.savez(
        heatmap_file(trials, scoring),
        win=data.win,
        tie=data.tie,
        score_diff=data.score_diff,
        game_len=data.game_len,
        rounds=data.rounds,
    )

def load_data(trials: int, scoring: str) -> HeatmapData:
    hm = np.load(heatmap_file(trials, scoring))
    return HeatmapData(
        win=hm['win'],
        tie=hm['tie'],
        score_diff=hm['score_diff'],
        game_len=hm['game_len'],
        rounds=hm['rounds'],
    )
