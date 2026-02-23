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
    DEBUG = args.debug

    # --- ALL POSSIBLE SEQUENCES ---
    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(s) for s in sequences]

    # --- PHASE 1: LOAD OR GENERATE DECKS ---
    # Do already scored decks of the given number of trials exist
    results_exist = data_exists(args.trials, args.scoring)

    # Do the number of decks already exist
    decks_exist_flag = decks_exists(args.trials)

    # If scored decks already exist, load then
    if results_exist and not args.regen:
        print(f'Loading matchup results for {args.trials} trials')
        data = load_data(args.trials, args.scoring)
    else:
        # Either no results or --regen
        # If the raw decks already exist, load them
        if decks_exist_flag:
            print(f'Loading existing decks for {args.trials} trials')
            decks = load_decks(args.trials)
        else:
            # Otherwise generate the decks
            print(f'Generating {args.trials} random decks')
            generate_and_save_decks(args.trials, regen=True)
            decks = load_decks(args.trials)

        # Score the raw decks for the number of trials
        print(f'Running all matchups for {args.trials} trials')
        data = run_all_matchups_from_decks(decks, sequences, args.scoring)
        if DEBUG:
            print('Debug output:')
            print(f'Decks: {decks}')
            print('Scored Dictionary - p1_win, p2_win, tie, score_diff, game_length, round_sum:')
            for key, value in data.items():
                print(f'{key}: {value}')
        save_data(data, args.trials, args.scoring)

    # --- OPTIONAL VISUALIZATION ---
    if args.no_plots:
        print('Skipping plot generation (--no-plots specified)')
    else:
        # plot_score_diff(data, labels, args.trials, args.scoring)
        # plot_score_diff_per_round(data, labels, args.trials, args.scoring)
        # plot_win_vs_score_diff(data, args.trials, args.scoring)
        plot_dominance_graph(data, labels, args.trials, args.scoring)
        plot_win_probability(data, labels, args.trials, args.scoring)


if __name__ == '__main__':
    main()

