import random
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- Game logic ---

def simulate_rons_game(seq1, seq2):
    """
    Simulate one full game of Ron's Game.
    Returns:
        1 if player 1 wins
        2 if player 2 wins
        0 if tie
    """
    # Create and shuffle deck
    deck = ['R'] * 26 + ['B'] * 26
    random.shuffle(deck)

    score1 = 0
    score2 = 0

    i = 0
    n = len(deck)

    while i < n:
        window = []
        cards_flipped = 0

        while i < n:
            window.append(deck[i])
            cards_flipped += 1
            i += 1

            if len(window) > 3:
                window.pop(0)

            if len(window) == 3:
                if tuple(window) == seq1:
                    score1 += cards_flipped
                    break
                elif tuple(window) == seq2:
                    score2 += cards_flipped
                    break

        # if fewer than 3 cards remain, no one can score
        if n - i < 3:
            break

    if score1 > score2:
        return 1
    elif score2 > score1:
        return 2
    else:
        return 0


# --- Tournament simulation ---

def run_simulation(seq1, seq2, trials=5000):
    """
    Returns Player 2 win percentage (ties ignored).
    """
    p2_wins = 0
    total_games = 0

    for _ in range(trials):
        result = simulate_rons_game(seq1, seq2)
        if result != 0:
            total_games += 1
            if result == 2:
                p2_wins += 1

    return p2_wins / total_games if total_games > 0 else 0.0


# --- Run all matchups ---

sequences = list(itertools.product(['R', 'B'], repeat=3))
labels = [''.join(seq) for seq in sequences]

heatmap = np.zeros((8, 8))

TRIALS = 5000

for i, seq1 in enumerate(sequences):
    for j, seq2 in enumerate(sequences):
        if seq1 == seq2:
            heatmap[i, j] = np.nan
        else:
            heatmap[i, j] = run_simulation(seq1, seq2, TRIALS)


# --- Plot heatmap ---

mask = np.triu(np.ones_like(heatmap, dtype=bool))

plt.figure(figsize=(10, 8))
sns.heatmap(
    heatmap,
    xticklabels=labels,
    yticklabels=labels,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=0,
    vmax=1,
    cbar_kws={"label": "Player 2 Win Percentage"}
)

plt.xlabel("Player 2 Sequence")
plt.ylabel("Player 1 Sequence")
plt.title("Ron's Game: Player 2 Winning Percentage")
plt.tight_layout()
plt.show()
