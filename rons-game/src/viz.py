import os
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np

FIG_DIR = 'figures'
os.makedirs(FIG_DIR, exist_ok=True)


# -------------------------
# Helpers
# -------------------------

def _matrix_from_data(
        data: dict, 
        labels: list[str], 
        value_fn
    ) -> np.ndarray[tuple[int, int], np.any]:
    '''
    Build NxN matrix from dict[(seq1, seq2)] -> tuple,
    applying value_fn to each tuple to extract desired value.
    Row = seq1, Column = seq2
    '''
    n = len(labels)
    mat = np.full((n, n), np.nan)
    for i, s1 in enumerate(labels):
        for j, s2 in enumerate(labels):
            key = (s1, s2)
            if key in data:
                mat[i, j] = value_fn(data[key])
    return mat.T


def lower_triangle_mask(mat):
    '''
    Return mask for lower triangle including diagonal
    '''
    return np.tri(mat.shape[0], k=0, dtype=bool)


# -------------------------
# Score Difference Heatmap
# -------------------------
def plot_score_diff(data: dict, labels: list[str], trials: int, scoring: str) -> None:
    score_diff = _matrix_from_data(data, labels, lambda v: v[3])
    game_len = _matrix_from_data(data, labels, lambda v: v[4])
    rounds = _matrix_from_data(data, labels, lambda v: v[5])

    mask = lower_triangle_mask(score_diff)
    annot = np.empty(score_diff.shape, dtype=object)

    for i in range(len(labels)):
        for j in range(len(labels)):
            annot[i, j] = '' if mask[i, j] else (
                f'{score_diff[i,j]:+.1f}\n({game_len[i,j]:.1f} | {rounds[i,j]:.1f})'
            )

    plt.figure(figsize=(11, 7))
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
    )

    plt.title('Score Differential\n(Game Length | Rounds)')
    plt.xlabel('Player 2 Sequence')
    plt.ylabel('Player 1 Sequence')
    os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_score_diff_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------
# Score Difference per Round
# -------------------------
def plot_score_diff_per_round(data: dict, labels: list[str], trials: int, scoring: str) -> None:
    score_diff = _matrix_from_data(data, labels, lambda v: v[3])
    rounds = _matrix_from_data(data, labels, lambda v: v[5])
    score_per_round = score_diff / np.maximum(rounds, 1e-9)

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
        cbar_kws={'label': 'Score Diff per Round (seq1 - seq2)'},
    )

    plt.title('Average Score Differential per Round')
    plt.xlabel('Player 2 Sequence')
    plt.ylabel('Player 1 Sequence')
    os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_score_diff_per_round_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------
# Win Probability vs Score Diff
# -------------------------
def plot_win_vs_score_diff(data: dict, trials: int, scoring: str) -> None:
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

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(xs, ys, c=colors, cmap='viridis', alpha=0.75)
    plt.axhline(0, linestyle='--', color='black')
    plt.axvline(0.5, linestyle='--', color='black')

    plt.colorbar(scatter, label='Avg Rounds')
    plt.xlabel('Player 1 Win Probability')
    plt.ylabel('Avg Score Differential (seq1 - seq2)')
    plt.title(f'Win Probability vs Score Differential ({scoring}, {trials:,})')
    plt.tight_layout()
    os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_win_vs_score_diff_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------
# Dominance Graph
# -------------------------
def plot_dominance_graph(data: dict, labels: list[str], trials: int, scoring: str) -> None:
    G = nx.DiGraph()
    G.add_nodes_from(labels)

    for i, s1 in enumerate(labels):
        for j, s2 in enumerate(labels):
            if i != j:
                p1_win = data[(s1, s2)][0]
                score_diff = data[(s1, s2)][3]
                if p1_win > 0.5 and score_diff < 0:
                    G.add_edge(s1, s2)  # loser -> winner

    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        nx.circular_layout(G),
        with_labels=True,
        node_size=2000,
        node_color='lightblue',
        arrowsize=15,
    )

    plt.title(f'Strategy Dominance Graph ({scoring}, {trials})')
    os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_dominance_graph_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()

# -------------------------
# Penney-Style Dominance Graph (Best Response)
# -------------------------

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def plot_penney_graph(data:dict, trials:int, scoring:str):

    sequences = sorted(list(set([k[0] for k in data.keys()])))
    
    # Find best response to each sequence
    best_response = {}

    for p1 in sequences:
        best_seq = None
        best_prob = -1

        for p2 in sequences:
            if p1 == p2:
                continue

            p1_win, p2_win, *_ = data[(p1, p2)]

            if np.isnan(p2_win):
                continue

            if p2_win > best_prob:
                best_prob = p2_win
                best_seq = p2

        best_response[p1] = (best_seq, best_prob)

    # Build graph
    G = nx.DiGraph()

    for seq in sequences:
        G.add_node(seq)

    for p1, (p2, prob) in best_response.items():
        G.add_edge(p1, p2, weight=prob)

    # Fixed circular layout
    pos = nx.circular_layout(G)

    plt.figure(figsize=(8,8))

    nx.draw_networkx_nodes(
        G, pos,
        node_size=3000,
        node_color='lightblue',
        edgecolors='black'
    )

    nx.draw_networkx_labels(
        G, pos,
        font_size=12,
        font_weight='bold'
    )

    nx.draw_networkx_edges(
        G, pos,
        arrows=True,
        arrowstyle='->',
        arrowsize=20,
        width=2,
        min_target_margin=30
    )

    # Edge labels
    edge_labels = {(u,v): f'{d['weight']:.2f}' for u,v,d in G.edges(data=True)}

    nx.draw_networkx_edge_labels(
        G, pos,
        edge_labels=edge_labels,
        font_size=10,
        label_pos=0.6
    )

    plt.title('Penney\'s Game - Best Counter Strategy')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_best_response_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()


# -------------------------
# Win Probability Heatmap
# -------------------------
def plot_win_probability(data: dict, labels: list[str], trials: int, scoring: str) -> None:
    '''
    Rows = seq1 (Player 1), Columns = seq2 (Player 2)
    '''
    win_prob = _matrix_from_data(data, labels, lambda v: v[0]) # P1 win prob
    tie_prob = _matrix_from_data(data, labels, lambda v: v[2])

    # mask = np.eye(win_prob.shape[0], dtype=bool)
    mask = lower_triangle_mask(win_prob).T


    annot = np.empty(win_prob.shape, dtype=object)
    for i in range(len(labels)):
        for j in range(len(labels)):
            if i == j or np.isnan(win_prob[i, j]):
                annot[i, j] = ''
            else:
                annot[i, j] = f'{win_prob[i,j]*100:.1f}%\n({tie_prob[i,j]*100:.1f}%)'

    plt.figure(figsize=(11, 7))
    sns.heatmap(
        win_prob,
        mask=mask,
        annot=annot,
        fmt='',
        cmap='coolwarm',
        center=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Player 1 Win Probability'},
    )

    plt.title(f'Player 2 Win Probability Heatmap ({scoring}, {trials:,} trials)')
    plt.xlabel('Player 2 Sequence')
    plt.ylabel('Player 1 Sequence')

    plt.tight_layout()
    os.makedirs(os.path.join(FIG_DIR, scoring), exist_ok=True)
    plt.savefig(f'{FIG_DIR}/{scoring}/rons_win_probability_{trials}.png', dpi=300, bbox_inches='tight')
    plt.show()

