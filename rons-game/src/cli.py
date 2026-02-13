import argparse

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

    parser.add_argument(
        '--scoring',
        choices=['cards', 'rounds'],
        default='cards',
        help='Scoring method: cards or rounds',
    )

    return parser.parse_args()
