import numpy as np
from tqdm import tqdm, trange
from src.data_types import HeatmapData, ScoringMode
from src.game import run_simulation, simulate_rons_game

def run_all_matchups_from_decks(decks, sequences, scoring):
    results = {}

    seq_strings = [''.join(seq) for seq in sequences]

    n = len(sequences)
    total = n * n

    with tqdm(total=total, desc="Scoring all matchups") as pbar:
        for i, s1 in enumerate(sequences):
            for j, s2 in enumerate(sequences):
                key = (seq_strings[i], seq_strings[j])

                if i == j:
                    results[key] = (np.nan,)*6
                else:
                    results[key] = score_matchup_from_decks(
                        decks, s1, s2, scoring
                    )

                pbar.update(1)

    return results


def score_matchup_from_decks(decks, seq1, seq2, scoring):
    trials = len(decks)

    p1 = p2 = ties = 0
    score_diff_sum = game_len_sum = rounds_sum = 0.0

    for i in trange(trials, leave=False):
        result, s1, s2, gl, r = simulate_rons_game(
             seq1, seq2, scoring, deck=decks[i]
        )

        score_diff_sum += (s2 - s1)
        game_len_sum += gl
        rounds_sum += r

        if result == 1:
            p1 += 1
        elif result == 2:
            p2 += 1
        else:
            ties += 1

    return (
        p1 / trials,
        p2 / trials,
        ties / trials,
        score_diff_sum / trials,
        game_len_sum / trials,
        rounds_sum / trials,
    )

