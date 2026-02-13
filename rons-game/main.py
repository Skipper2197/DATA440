import itertools
from src.cli import parse_args
from src.utils import data_exists
from src.in_out import load_data, save_data
from src.simulation import run_all_matchups
from src.viz import (
    plot_score_diff,
    plot_score_diff_per_round,
    plot_win_vs_score_diff,
    plot_dominance_graph,
    plot_win_probability,
)

def main() -> None:
    args = parse_args()

    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(s) for s in sequences]

    if data_exists(args.trials, args.scoring) and not args.regen:
        print(f'Loading data for {args.trials} trials')
        data = load_data(args.trials, args.scoring)
    else:
        print(f'Generating data for {args.trials} trials')
        data = run_all_matchups(sequences, args.trials, args.scoring)
        save_data(data, args.trials, args.scoring)

    if args.no_plots:
        print('Skipping plot generation (--no-plots specified)')
        return

    plot_score_diff(data, labels, args.trials, args.scoring)
    plot_score_diff_per_round(data, labels, args.trials, args.scoring)
    plot_win_vs_score_diff(data, args.trials, args.scoring)
    plot_dominance_graph(data, labels, args.trials, args.scoring)
    plot_win_probability(data, labels, args.trials, args.scoring)

if __name__ == "__main__":
    main()
