import itertools
from src.cli import parse_args
from src.generate_decks import generate_and_save_decks
from src.utils import ensure_project_dirs
from src.in_out import load_data, save_data, load_decks, decks_exists, data_exists
from src.simulation import run_all_matchups_from_decks
from src.viz import (
    plot_score_diff,
    plot_score_diff_per_round,
    plot_win_vs_score_diff,
    plot_dominance_graph,
    plot_win_probability,
)

def main() -> None:
    ensure_project_dirs()
    args = parse_args()

    # --- ALL POSSIBLE SEQUENCES ---
    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(s) for s in sequences]

    # --- PHASE 1: LOAD OR GENERATE DECKS ---
    results_exist = data_exists(args.trials, args.scoring)
    decks_exist_flag = decks_exists(args.trials)

    if results_exist and not args.regen:
        print(f"Loading matchup results for {args.trials} trials")
        data = load_data(args.trials, args.scoring)
    else:
        # Either no results or --regen
        if decks_exist_flag:
            print(f"Loading existing decks for {args.trials} trials")
            decks = load_decks(args.trials)
        else:
            print(f"Generating {args.trials} random decks")
            generate_and_save_decks(args.trials, overwrite=True)
            decks = load_decks(args.trials)

        print(f"Running all matchups for {args.trials} trials")
        data = run_all_matchups_from_decks(decks, sequences, args.scoring)
        save_data(data, args.trials, args.scoring)

    # --- OPTIONAL VISUALIZATION ---
    if args.no_plots:
        print("Skipping plot generation (--no-plots specified)")
    else:
        # plot_score_diff(data, labels, args.trials, args.scoring)
        # plot_score_diff_per_round(data, labels, args.trials, args.scoring)
        plot_win_vs_score_diff(data, args.trials, args.scoring)
        plot_dominance_graph(data, labels, args.trials, args.scoring)
        plot_win_probability(data, labels, args.trials, args.scoring)


if __name__ == "__main__":
    main()

