import numpy as np
from tqdm import tqdm
from src.data_types import HeatmapData, ScoringMode
from src.game import run_simulation

def run_all_matchups(
    sequences: list[tuple[str, ...]],
    trials: int,
    scoring: ScoringMode = "cards"
) -> HeatmapData:

    n = len(sequences)
    data = HeatmapData(
        win=np.full((n, n), np.nan),
        tie=np.full((n, n), np.nan),
        score_diff=np.full((n, n), np.nan),
        game_len=np.full((n, n), np.nan),
        rounds=np.full((n, n), np.nan),
    )

    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]

    for i, j in tqdm(pairs, desc='Matchups', total=len(pairs)):
        _, p2, ties, diff, gl, r = run_simulation(
            sequences[i], sequences[j], trials, scoring
        )

        data.win[i, j] = p2
        data.tie[i, j] = ties
        data.score_diff[i, j] = diff
        data.game_len[i, j] = gl
        data.rounds[i, j] = r

    return data
