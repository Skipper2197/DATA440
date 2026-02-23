import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

FIG_DIR = "figures"
os.makedirs(FIG_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------

def _matrix_from_data(data: dict, labels: list[str], value_fn):
    """
    Build NxN matrix from dict[(seq1, seq2)] -> tuple,
    applying value_fn to each tuple to extract desired value.
    Row = seq1, Column = seq2
    """
    n = len(labels)
    mat = np.full((n, n), np.nan)
    for i, s1 in enumerate(labels):
        for j, s2 in enumerate(labels):
            key = (s1, s2)
            if key in data:
                mat[i, j] = value_fn(data[key])
    return mat.T


def lower_triangle_mask(mat):
    """Return mask for lower triangle including diagonal"""
    return np.tri(mat.shape[0], k=0, dtype=bool)


# -------------------------
# Score Difference Heatmap
# -------------------------
def plot_score_diff(ax, data: dict, labels: list[str], trials: int, scoring: str) -> None:
    score_diff = _matrix_from_data(data, labels, lambda v: v[3])
    game_len = _matrix_from_data(data, labels, lambda v: v[4])
    rounds = _matrix_from_data(data, labels, lambda v: v[5])

    mask = lower_triangle_mask(score_diff).T
    annot = np.empty(score_diff.shape, dtype=object)

    for i in range(len(labels)):
        for j in range(len(labels)):
            annot[i, j] = '' if mask[i, j] else (
                f'{score_diff[i,j]:+.1f}\n({game_len[i,j]:.1f} | {rounds[i,j]:.1f})'
            )

    ax.clear()
    sns.heatmap(
        score_diff,
        mask=mask,
        annot=annot,
        fmt='',
        cmap='coolwarm',
        center=0,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Avg Score Diff (seq1 - seq2)'},
        ax=ax
    )

    ax.set_title('Score Differential\n(Game Length | Rounds)')
    ax.set_xlabel('Player 2 Sequence')
    ax.set_ylabel('Player 1 Sequence')
    # os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    # plt.savefig(f'{FIG_DIR}/{scoring}/rons_score_diff_{trials}.png', dpi=300, bbox_inches='tight')


# -------------------------
# Score Difference per Round
# -------------------------
def plot_score_diff_per_round(ax, data: dict, labels: list[str], trials: int, scoring: str) -> None:
    score_diff = _matrix_from_data(data, labels, lambda v: v[3])
    rounds = _matrix_from_data(data, labels, lambda v: v[5])
    score_per_round = score_diff / np.maximum(rounds, 1e-9)

    mask = lower_triangle_mask(score_per_round).T

    ax.clear()
    sns.heatmap(
        score_per_round,
        mask=mask,
        cmap='coolwarm',
        center=0,
        annot=True,
        fmt='+.2f',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Score Diff per Round (seq1 - seq2)'},
        ax=ax
    )

    ax.set_title('Average Score Differential per Round')
    ax.set_xlabel('Player 2 Sequence')
    ax.set_ylabel('Player 1 Sequence')
    # os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    # plt.savefig(f'{FIG_DIR}/{scoring}/rons_score_diff_per_round_{trials}.png', dpi=300, bbox_inches='tight')


# -------------------------
# Win Probability vs Score Diff
# -------------------------
def plot_win_vs_score_diff(ax, data: dict, trials: int, scoring: str) -> None:
    labels = sorted(list(set(k[0] for k in data.keys())))
    xs, ys, colors = [], [], []

    for i, s1 in enumerate(labels):
        for j, s2 in enumerate(labels):
            if i != j:
                # Player 1 win probability (row=seq1, col=seq2)
                p1_win = data[(s1, s2)][0]
                score_diff = data[(s1, s2)][3]
                rounds = data[(s1, s2)][5]

                xs.append(p1_win)  # Player 1 win prob
                ys.append(score_diff)
                colors.append(rounds)

    ax.clear()
    scatter = ax.scatter(xs, ys, c=colors, cmap='viridis', alpha=0.75)
    ax.axhline(0, linestyle='--', color='black')
    ax.axvline(0.5, linestyle='--', color='black')

    ax.set_xlabel('Player 1 Win Probability')
    ax.set_ylabel('Avg Score Differential (seq1 - seq2)')
    ax.set_title(f'Win Probability vs Score Differential ({scoring}, {trials:,})')
    ax.figure.colorbar(scatter, ax=ax, label='Avg Rounds')
    # plt.tight_layout()
    # os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    # plt.savefig(f'{FIG_DIR}/{scoring}/rons_win_vs_score_diff_{trials}.png', dpi=300, bbox_inches='tight')


# -------------------------
# Dominance Graph
# -------------------------
def plot_dominance_graph(ax, data: dict, labels: list[str], trials: int, scoring: str) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i, s1 in enumerate(labels):
        for j, s2 in enumerate(labels):
            if i != j:
                p1_win = data[(s1, s2)][0]
                score_diff = data[(s1, s2)][3]
                if p1_win > 0.5 and score_diff < 0:
                    G.add_edge(s1, s2)  # winner -> loser

    ax.clear()
    nx.draw(
        G,
        nx.circular_layout(G),
        with_labels=True,
        node_size=2000,
        node_color='lightblue',
        arrowsize=15,
        ax=ax
    )

    ax.set_title(f'Strategy Dominance Graph ({scoring}, {trials})')
    # os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    # plt.savefig(f'{FIG_DIR}/{scoring}/rons_dominance_graph_{trials}.png', dpi=300, bbox_inches='tight')


# -------------------------
# Win Probability Heatmap
# -------------------------
def plot_win_probability(ax, data: dict, labels: list[str], trials: int, scoring: str) -> None:
    """
    Rows = seq1 (Player 1), Columns = seq2 (Player 2)
    """
    win_prob = _matrix_from_data(data, labels, lambda v: v[0]) # P1 win prob
    tie_prob = _matrix_from_data(data, labels, lambda v: v[2])

    # mask = np.eye(win_prob.shape[0], dtype=bool)
    mask = lower_triangle_mask(win_prob).T


    annot = np.empty(win_prob.shape, dtype=object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or np.isnan(win_prob[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = f"{win_prob[i,j]*100:.1f}%\n({tie_prob[i,j]*100:.1f}%)"

    ax.clear()
    sns.heatmap(
        win_prob,
        mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Player 1 Win Probability"},
        ax=ax
    )

    ax.set_title(f"Player 2 Win Probability Heatmap ({scoring}, {trials:,} trials)")
    ax.set_xlabel("Player 2 Sequence")
    ax.set_ylabel("Player 1 Sequence")

    #plt.tight_layout()
    #os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    #plt.savefig(f"{FIG_DIR}/{scoring}/rons_win_probability_{trials}.png", dpi=300, bbox_inches='tight')
    #plt.show()

