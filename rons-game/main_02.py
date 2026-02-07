import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from numpy.typing import NDArray

# Game logic

def simulate_rons_game_fast(seq1: tuple[str, ...], 
                            seq2: tuple[str, ...]) -> tuple[Literal[1, 2, 0], int, int]:
    # Create random deck
    deck = np.random.choice([0,1], size=52)
    s1 = tuple(1 if c == 'R' else 0 for c in seq1)
    s2 = tuple(1 if c == 'R' else 0 for c in seq2)

    score1 = score2 = 0
    i = 0
    num_rounds = 0

    # Loop through deck
    while i < 52:
        window = []
        cards = 0

        # Loop through possible scoring sequence
        while i < 52:
            window.append(deck[i])
            cards += 1
            i += 1

            # Only track last three cards
            if len(window) > 3:
                window.pop(0)

            if len(window) == 3:
                t = tuple(window)
                # If player 1 wins
                if t == s1:
                    score1 += cards
                    num_rounds += 1
                    break
                # If player 2 wins
                elif t == s2:
                    score2 += cards
                    num_rounds += 1
                    break
        # Break if there are fewer than 3 cards remaining
        if 52 - i < 3:
            break

    return (1 if score1 > score2 else 2 if score2 > score1 else 0, score1, score2, 
            i, # game length
            num_rounds
            )

# Tournament simulation
def run_simulation(seq1: tuple[str, ...], 
                   seq2: tuple[str, ...], 
                   trials: int=5000) -> tuple[float, float, float, float]:
                   
    p1 = p2 = ties = 0
    score_diff_sum = 0.0
    game_length_sum = 0.0
    rounds_sum = 0.0

    # Loop through trials
    for _ in range(trials):
        # Get result from game
        result, score1, score2, length, rounds = simulate_rons_game_fast(seq1, seq2)
        game_length_sum += length
        rounds_sum += rounds

        # Track score
        score_diff_sum += (score2 - score1)

        # Check for loss/win/tie
        if result == 1:
            p1 += 1
        elif result == 2:
            p2 += 1
        else:
            ties += 1

    return p1/trials, p2/trials, ties/trials, score_diff_sum/trials, game_length_sum/trials, rounds_sum/trials

# Run all matchups
def run_all_matchups(sequences: list[tuple[str, ...]],
                     win_heatmap: NDArray[np.float64], 
                     tie_heatmap: NDArray[np.float64],
                     score_diff_heatmap: NDArray[np.float64],
                     game_len_heatmap: NDArray[np.float64],
                     rounds_heatmap: NDArray[np.float64],
                     trials: int) -> None:
    for i, seq1 in enumerate(sequences):
        for j, seq2 in enumerate(sequences):
            if i == j:
                win_heatmap[i, j] = np.nan
                tie_heatmap[i, j] = np.nan
                score_diff_heatmap[i, j] = np.nan
                game_len_heatmap[i, j] = np.nan
                rounds_heatmap[i, j] = np.nan
            else:
                _, p2, ties, diff, game_len, rounds = run_simulation(seq1, seq2, trials)
                win_heatmap[i, j] = p2
                tie_heatmap[i, j] = ties
                score_diff_heatmap[i, j] = diff
                game_len_heatmap[i, j] = game_len
                rounds_heatmap[i, j] = rounds
    return

# Plot heatmaps side by side
def plot_wins_ties(win_heatmap: NDArray[np.float64], 
                   tie_heatmap: NDArray[np.float64], 
                   labels: list[str], 
                   trials: int) -> None:
    
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Ignore the top half of the heatmap
    mask = np.triu(np.ones_like(win_heatmap, dtype=bool))

    # Wins heatmap
    sns.heatmap(
        win_heatmap,
        ax=axes[0],
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        annot=True,
        fmt=".3f",
        cmap="coolwarm",
        center=0.5,
        cbar_kws={"label": "Player 2 Win Probability"}
    )

    # Wins heatmap labels
    axes[0].set_title("Player 2 Win Probability")
    axes[0].set_xlabel("Player 2 Sequence")
    axes[0].set_ylabel("Player 1 Sequence")

    # Ties heatmap
    sns.heatmap(
        tie_heatmap,
        ax=axes[1],
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        annot=True,
        fmt=".4f",
        cmap="Greys",
        cbar_kws={"label": "Tie Probability"}
    )

    # Ties heatmap labels
    axes[1].set_title("Tie Probability")
    axes[1].set_xlabel("Player 2 Sequence")
    axes[1].set_ylabel("")

    plt.tight_layout()

    # Save into figures folder
    filename = f"figures/rons_game_heatmaps_trials_{trials}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")

    plt.show()

    return

def plot_score_diff(score_diff_heatmap: NDArray[np.float64],
                    game_len_heatmap: NDArray[np.float64],
                    rounds_heatmap: NDArray[np.float64], 
                    labels: list[str], 
                    trials: int) -> None:

    plt.figure(figsize=(11, 7))

    # Ignore top half of heatmap
    mask = np.triu(np.ones_like(score_diff_heatmap, dtype=bool))

    # Build annotation matrix (strings)
    annot = np.empty(score_diff_heatmap.shape, dtype=object)

    for i in range(score_diff_heatmap.shape[0]):
        for j in range(score_diff_heatmap.shape[1]):
            if mask[i, j] or np.isnan(score_diff_heatmap[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = (
                    f"{score_diff_heatmap[i, j]:+.1f} \n"  
                    f"({game_len_heatmap[i, j]:.1f} | {rounds_heatmap[i, j]:.1f})"
                )

    # Make heatmap
    sns.heatmap(
        score_diff_heatmap,
        xticklabels=labels,
        yticklabels=labels,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0,
        cbar_kws={"label": "Average Score Difference (P2 - P1)"}
    )

    # Labels
    plt.title(
        "Average Score Differential\n"
        "(Avg Game Length | Avg Rounds per Game)"
    )
    plt.xlabel("Player 2 Sequence")
    plt.ylabel("Player 1 Sequence")
    plt.tight_layout()

    # Save into figures folder
    filename = f"figures/rons_game_score_diff_length_rounds_trials_{trials}.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    
    plt.show()

    return

def main() -> None:
    TRIALS = 100_000

    # Create a list of all possible combinations
    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(seq) for seq in sequences]

    # Heatmap tracking variables
    win_heatmap = np.zeros((8, 8))
    tie_heatmap = np.zeros((8, 8))
    score_diff_heatmap = np.zeros((8, 8))
    game_len_heatmap = np.zeros((8,8))
    rounds_heatmap = np.zeros((8, 8))

    # Run game and plot results
    run_all_matchups(sequences, win_heatmap, tie_heatmap, score_diff_heatmap, game_len_heatmap, rounds_heatmap, trials=TRIALS)
    # plot_wins_ties(win_heatmap, tie_heatmap, labels, trials=TRIALS)
    plot_score_diff(score_diff_heatmap, game_len_heatmap, rounds_heatmap, labels, trials=TRIALS)

    return

if __name__ == '__main__':
    main()
