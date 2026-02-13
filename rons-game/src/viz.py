import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
from src.data_types import HeatmapData
from src.utils import lower_triangle_mask

def plot_score_diff(data: HeatmapData, labels: list[str], trials: int, scoring: str) -> None:

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
        f'figures/{scoring}/rons_score_diff_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_score_diff_per_round(data: HeatmapData, labels: list[str], trials: int, scoring: str) -> None:

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
        f'figures/{scoring}/rons_score_diff_per_round_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_win_vs_score_diff(data: HeatmapData, trials: int, scoring: str) -> None:

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
        f'figures/{scoring}/rons_win_vs_score_diff_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_dominance_graph(data: HeatmapData, labels: list[str], trials: int, scoring: str) -> None:

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
        f'figures/{scoring}/rons_dominance_graph_{trials}.png',
        dpi=300,
        bbox_inches='tight',
    )
    plt.show()

def plot_win_probability(data: HeatmapData, labels: list[str], trials: int, scoring: str) -> None:
    """
    Plot Player 2 win probability heatmap.
    """

    # MASK GENERATION
    # mask = lower_triangle_mask(data.win)

    # annot = np.empty(data.win.shape, dtype=object)

    # for i in range(data.win.shape[0]):
    #     for j in range(data.win.shape[1]):
    #         if mask[i, j] or np.isnan(data.win[i, j]):
    #             annot[i, j] = ""
    #         else:
    #             annot[i, j] = (
    #                 f"{data.win[i, j]:.3f}\n"
    #                 f"({data.tie[i, j]:.3f})"
    #             )

    # NO MASK GENERATION - CHECK FOR CONVERGENCE
    annot = np.empty(data.win.shape, dtype=object)

    for i in range(data.win.shape[0]):
        for j in range(data.win.shape[1]):
            if i == j or np.isnan(data.win[i, j]):
                annot[i, j] = ""
            else:
                annot[i, j] = (
                    f"{data.win[i, j]:.3f}\n"
                    f"({data.tie[i, j]:.3f})"
                )

    plt.figure(figsize=(11, 7))
    sns.heatmap(
        data.win,
        # mask=mask,
        annot=annot,
        fmt="",
        cmap="coolwarm",
        center=0.5,
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={"label": "Player 2 Win Probability"},
    )

    plt.title("Player 2 Win Probability Heatmap")
    plt.xlabel("Player 2 Sequence")
    plt.ylabel("Player 1 Sequence")
    plt.tight_layout()

    plt.savefig(
        f"figures/{scoring}/rons_win_probability_{trials}.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show()

    max_asym = np.nanmax(
        np.abs(
            data.win + data.win.T + data.tie - 1
        )
    )
    print(f"Max antisymmetry error: {max_asym:.4e}")