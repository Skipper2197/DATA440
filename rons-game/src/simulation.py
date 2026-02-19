import numpy as np
from tqdm import tqdm, trange
from src.game import simulate_rons_game

def run_all_matchups_from_decks(
        decks: np.ndarray,
        sequences: list[tuple[str, ...]],
        scoring: str
    ) -> dict:
    
    '''
    Docstring for run_all_matchups_from_decks
    
    :param decks: raw decks to be scored
    :type decks: np.ndarray
    :param sequences: all matchup sequences
    :type sequences: list[tuple[str, ...]]
    :param scoring: cards or rounds
    :type scoring: str
    :return: dictionary of results with key of (seq1, seq2)
    :rtype: dict
    '''

    results = {}

    seq_strings = [''.join(seq) for seq in sequences]

    n = len(sequences)
    total = n * n

    with tqdm(total=total, desc='Scoring all matchups') as pbar:
        for i, s1 in enumerate(sequences):
            for j, s2 in enumerate(sequences):
                key = (seq_strings[i], seq_strings[j])
                # Diagonal matchups do not make sense so store as nan
                if i == j:
                    results[key] = (np.nan,)*6
                else:
                    # at key (s1,s2) get all stats
                    results[key] = score_matchup_from_decks(
                        decks, s1, s2, scoring
                    )

                pbar.update(1)

    return results


def score_matchup_from_decks(
        decks: np.ndarray,
        seq1: tuple[str, ...],
        seq2: tuple[str, ...],
        scoring: str
    ) -> tuple[float, float, float, float, float, float]:
    
    '''
    Docstring for score_matchup_from_decks
    
    :param decks: raw decks to be scored
    :type decks: np.ndarray
    :param seq1: player 1 sequence
    :type seq1: tuple[str, ...]
    :param seq2: player 2 seuquence
    :type seq2: tuple[str, ...]
    :param scoring: cards or rounds
    :type scoring: str
    :return: tuple of p1 win percent, p2 win percent, tie percent, and other metrics
    :rtype: tuple[float, float, float, float, float, float]
    '''

    trials = len(decks)

    p1 = p2 = ties = 0
    score_diff_sum = game_len_sum = rounds_sum = 0.0

    for i in trange(trials, leave=False):
        # Simulate game
        result, s1, s2, gl, r = simulate_rons_game(
             seq1, seq2, scoring, deck=decks[i]
        )

        # Update statistics
        score_diff_sum += (s2 - s1)
        game_len_sum += gl
        rounds_sum += r

        # track win or tie
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
