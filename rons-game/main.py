import itertools
import time
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

    start_total = time.time()

    # --- ALL POSSIBLE SEQUENCES ---
    start_sequences = time.time()
    sequences = list(itertools.product(['R', 'B'], repeat=3))
    labels = [''.join(s) for s in sequences]
    print(f'Generated {len(labels)} sequences in {time.time() - start_sequences:.2f}s')

    # --- PHASE 1: LOAD OR GENERATE DECKS ---
    start_phase1 = time.time()
    results_exist = data_exists(args.trials, args.scoring)
    decks_exist_flag = decks_exists(args.trials)

    if results_exist and not args.regen:
        start_load = time.time()
        print(f'Loading matchup results for {args.trials} trials')
        data = load_data(args.trials, args.scoring)
        print(f'Loaded matchup results in {time.time() - start_load:.2f}s')
    else:
        if decks_exist_flag:
            start_load_decks = time.time()
            print(f'Loading existing decks for {args.trials} trials')
            decks = load_decks(args.trials)
            print(f'Loaded decks in {time.time() - start_load_decks:.2f}s')
        else:
            start_gen_decks = time.time()
            print(f'Generating {args.trials} random decks')
            generate_and_save_decks(args.trials, regen=True)
            decks = load_decks(args.trials)
            print(f'Generated and loaded decks in {time.time() - start_gen_decks:.2f}s')

        start_matchups = time.time()
        print(f'Running all matchups for {args.trials} trials')
        data = run_all_matchups_from_decks(decks, sequences, args.scoring)
        print(f'Completed all matchups in {time.time() - start_matchups:.2f}s')

        if DEBUG:
            print('Debug output:')
            print(f'Decks: {decks}')
            print('Scored Dictionary - p1_win, p2_win, tie, score_diff, game_length, round_sum:')
            for key, value in data.items():
                print(f'{key}: {value}')

        start_save = time.time()
        save_data(data, args.trials, args.scoring)
        print(f'Saved matchup data in {time.time() - start_save:.2f}s')

    print(f'Phase 1 completed in {time.time() - start_phase1:.2f}s')

    # --- OPTIONAL VISUALIZATION ---
    start_viz = time.time()
    if args.no_plots:
        print('Skipping plot generation (--no-plots specified)')
        print(f'Total execution time: {time.time() - start_total:.2f}s')
    else:
        print(f'Total execution time: {time.time() - start_total:.2f}s')
        plot_dominance_graph(data, labels, args.trials, args.scoring)
        plot_win_probability(data, labels, args.trials, args.scoring)

if __name__ == '__main__':
    main()

