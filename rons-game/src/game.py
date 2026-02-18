import numpy as np
import os
import numpy as np
from tqdm import trange
from typing import Literal
from src.data_types import ScoringMode

def simulate_rons_game(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
    scoring: ScoringMode = "cards",
    deck: np.ndarray | None = None
) -> tuple[Literal[1, 2, 0], int, int, int, int]:

    # STORE DECK HERE - APPEND TO END OF DECK FILE

    s1 = tuple(1 if c == 'R' else 0 for c in seq1)
    s2 = tuple(1 if c == 'R' else 0 for c in seq2)

    score1 = score2 = 0
    i = 0
    num_rounds = 0

    while i < 52:
        window = []
        cards = 0

        while i < 52:
            window.append(deck[i])
            cards += 1
            i += 1

            if len(window) > 3:
                window.pop(0)

            if len(window) == 3:
                t = tuple(window)
                if t == s1:
                    score1 += cards if scoring == "cards" else 1
                    num_rounds += 1
                    break
                elif t == s2:
                    score2 += cards if scoring == "cards" else 1
                    num_rounds += 1
                    break

        if 52 - i < 3:
            break

    winner = 1 if score1 > score2 else 2 if score2 > score1 else 0
    return winner, score1, score2, i, num_rounds


def run_simulation(
    deck_file: str,
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
    scoring: ScoringMode = "cards"
) -> tuple[float, float, float, float, float, float]:

    p1 = p2 = ties = 0
    score_diff_sum = game_len_sum = rounds_sum = 0.0

    data = np.load(deck_file)
    decks = data['decks']

    trials = len(decks)

    for i in trange(
        trials,
        desc=f'{scoring} | {"".join(seq1)} vs {"".join(seq2)}',
        leave=False
    ):
        result, s1, s2, gl, r = simulate_rons_game(seq1, seq2, scoring, deck=decks[i])
        score_diff_sum += (s2 - s1)
        game_len_sum += gl
        rounds_sum += r

        if result == 1:
            p1 += 1
        elif result == 2:
            p2 += 1
        else:
            ties += 1

    result_array = np.array([
        p1 / trials,
        p2 / trials,
        ties / trials,
        score_diff_sum / trials,
        game_len_sum / trials,
        rounds_sum / trials,
    ])

    name = f"{scoring}_{''.join(seq1)}_vs_{''.join(seq2)}"
    out_path = os.path.join('data/results', f"{name}.npy")

    np.save(out_path, result_array)
    print(f"Saved results → {out_path}")

    return tuple(result_array)
