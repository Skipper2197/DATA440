import itertools
import os
from dataclasses import dataclass
from typing import Literal

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
from numpy.typing import NDArray

import argparse

# ====================================================================================================
# CLI argument parser
# ====================================================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate and analyze Ron's Game"
    )

    parser.add_argument(
        '--trials',
        type=int,
        default=100_000,
        help='Number of trials per matchup (default: 100000)',
    )

    parser.add_argument(
        '--regen',
        action='store_true',
        help='Force regeneration of simulation data even if cached',
    )

    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip plotting',
    )

    return parser.parse_args()


# ====================================================================================================
# Data container
# ====================================================================================================

@dataclass
class HeatmapData:
    win: NDArray[np.float64]
    tie: NDArray[np.float64]
    score_diff: NDArray[np.float64]
    game_len: NDArray[np.float64]
    rounds: NDArray[np.float64]

# ====================================================================================================
# Utilities
# ====================================================================================================

def heatmap_file(trials: int) -> str:
    return f'data/heatmaps_{trials}.npz'

def lower_triangle_mask(arr: NDArray[np.float64]) -> NDArray[np.bool_]:
    return np.triu(np.ones_like(arr, dtype=bool))

def data_exists(trials: int) -> bool:
    return os.path.exists(heatmap_file(trials))

# ====================================================================================================
# Game logic
# ====================================================================================================

def simulate_rons_game_fast(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
) -> tuple[Literal[1, 2, 0], int, int, int, int]:

    deck = np.random.choice([0, 1], size=52)
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
                    score1 += cards
                    num_rounds += 1
                    break
                elif t == s2:
                    score2 += cards
                    num_rounds += 1
                    break

        if 52 - i < 3:
            break

    winner = 1 if score1 > score2 else 2 if score2 > score1 else 0
    return winner, score1, score2, i, num_rounds

def run_simulation(
    seq1: tuple[str, ...],
    seq2: tuple[str, ...],
    trials: int,
) -> tuple[float, float, float, float, float, float]:

    p1 = p2 = ties = 0
    score_diff_sum = game_len_sum = rounds_sum = 0.0

    for _ in range(trials):
        result, s1, s2, gl, r = simulate_rons_game_fast(seq1, seq2)
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

# ====================================================================================================
# Simulation runner
# ====================================================================================================

def run_all_matchups(
    sequences: list[tuple[str, ...]],
    trials: int,
) -> HeatmapData:

    n = len(sequences)
    data = HeatmapData(
        win=np.full((n, n), np.nan),
        tie=np.full((n, n), np.nan),
        score_diff=np.full((n, n), np.nan),
        game_len=np.full((n, n), np.nan),
        rounds=np.full((n, n), np.nan),
    )

    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            if i == j:
                continue

            _, p2, ties, diff, gl, r = run_simulation(seq1, seq2, trials)

            data.win[i, j] = p2
            data.tie[i, j] = ties
            data.score_diff[i, j] = diff
            data.game_len[i, j] = gl
            data.rounds[i, j] = r

    return data

# ====================================================================================================
# Input/Output
# ====================================================================================================

def save_data(data: HeatmapData, trials: int) -> None:
    np.savez(
        heatmap_file(trials),
        win=data.win,
        tie=data.tie,
        score_diff=data.score_diff,
        game_len=data.game_len,
        rounds=data.rounds,
    )


def load_data(trials: int) -> HeatmapData:
    hm = np.load(heatmap_file(trials))
    return HeatmapData(
        win=hm['win'],
        tie=hm['tie'],
        score_diff=hm['score_diff'],
        game_len=hm['game_len'],
        rounds=hm['rounds'],
    )

# ====================================================================================================
# Plotting
# ====================================================================================================

def plot_score_diff(data: HeatmapData, labels: list[str], trials: int) -> None:

    mask = lower_triangle_mask(data.score_diff)
    annot = np.empty(data.score_diff.shape, dtype=object)

    for i in range(annot.shape[0]):
        for j in range(annot.shape[1]):
            annot[i, j] = '' if mask[i, j] else (
                f'{data.score_diff[i,j]:+.1f}\n'
                f'({data.game_len[i,j]:.1f} | {data.rounds[i,j]:.1f})'
            )

    plt.figure(figsize=(11, 7))
    sns.heatmap(
        data.score_diff,
        mask=mask,
        annot=annot,
        fmt='',
        cmap='coolwarm',
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Avg Score Diff (P2 - P1)'},
    )

    plt.title('Score Differential\n(Game Length | Rounds)')
    plt.xlabel('Player 2 Sequence')
    plt.ylabel('Player 1 Sequence')
    plt.tight_layout()

    plt.savefig(
        f'figures/rons_score_diff_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_score_diff_per_round(data: HeatmapData, labels: list[str], trials: int) -> None:

    score_per_round = data.score_diff / data.rounds
    mask = lower_triangle_mask(score_per_round)

    plt.figure(figsize=(11, 7))
    sns.heatmap(
        score_per_round,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='+.2f',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Score Diff per Round (P2 - P1)'},
    )

    plt.title('Average Score Differential per Round')
    plt.xlabel('Player 2 Sequence')
    plt.ylabel('Player 1 Sequence')
    plt.tight_layout()

    plt.savefig(
        f'figures/rons_score_diff_per_round_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_win_vs_score_diff(data: HeatmapData, trials: int) -> None:

    xs, ys, colors = [], [], []

    for i in range(8):
        for j in range(8):
            if i != j:
                xs.append(data.win[i, j])
                ys.append(data.score_diff[i, j])
                colors.append(data.rounds[i, j])

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=colors, cmap='viridis', alpha=0.75)
    plt.axhline(0, linestyle='--', color='black')
    plt.axvline(0.5, linestyle='--', color='black')

    plt.colorbar(scatter, label='Avg Rounds')
    plt.xlabel('Player 2 Win Probability')
    plt.ylabel('Avg Score Differential (P2 − P1)')
    plt.title('Win Probability vs Score Differential')
    plt.tight_layout()

    plt.savefig(
        f'figures/rons_win_vs_score_diff_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_dominance_graph(data: HeatmapData, labels: list[str], trials: int) -> None:

    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i in range(8):
        for j in range(8):
            if i != j and data.win[i, j] > 0.5 and data.score_diff[i, j] > 0:
                G.add_edge(labels[j], labels[i])

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        nx.circular_layout(G),
        with_labels=True,
        node_size=2000,
        node_color='lightblue',
        arrowsize=15,
    )

    plt.title('Strategy Dominance Graph')
    plt.savefig(
        f'figures/rons_dominance_graph_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

# ====================================================================================================
# Main
# ====================================================================================================

def main() -> None:

    cli_args = parse_args()

    TRIALS = cli_args.trials
    REGEN = cli_args.regen
    NO_PLOTS = cli_args.no_plots

    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(s) for s in sequences]

    if data_exists(TRIALS) and not REGEN:
        print(f'Loading data for {TRIALS} trials')
        data = load_data(TRIALS)
    else:
        print(f'Generating data for {TRIALS} trials')
        data = run_all_matchups(sequences, TRIALS)
        save_data(data, TRIALS)

    if NO_PLOTS:
        print('Skipping plot generation (--no-plots specified)')
    else:
        plot_score_diff(data, labels, TRIALS)
        plot_score_diff_per_round(data, labels, TRIALS)
        plot_win_vs_score_diff(data, TRIALS)
        plot_dominance_graph(data, labels, TRIALS)


if __name__ == '__main__':
    main()
