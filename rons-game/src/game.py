import numpy as np
import os
from tqdm import trange
from typing import Literal
from src.data_types import ScoringMode

def simulate_rons_game(
        seq1: tuple[str, ...],
        seq2: tuple[str, ...],
        scoring: ScoringMode = 'cards',
        deck: np.ndarray | None = None
    ) -> tuple[Literal[1, 2, 0], int, int, int, int]:

    '''
    Run through a 52 card deck and track statistics for the game
    
    :param seq1: Player 1 sequence of colors
    :type seq1: tuple[str, ...]
    :param seq2: Player 2 sequence of colors
    :type seq2: tuple[str, ...]
    :param scoring: cards or rounds
    :type scoring: ScoringMode
    :param deck: Random generated deck
    :type deck: np.ndarray | None
    :return: Statistics for one round of the game
    :rtype: tuple[Literal[1, 2, 0], int, int, int, int]
    '''

    s1 = tuple(1 if c == 'R' else 0 for c in seq1)
    s2 = tuple(1 if c == 'R' else 0 for c in seq2)

    score1 = score2 = 0
    i = 0
    num_rounds = 0

    while i < 52:
        window = []
        cards = 0

        while i < 52:
            window.append(deck[i])
            cards += 1
            i += 1

            if len(window) > 3:
                window.pop(0)

            if len(window) == 3:
                t = tuple(window)
                if t == s1:
                    score1 += cards if scoring == 'cards' else 1
                    num_rounds += 1
                    break
                elif t == s2:
                    score2 += cards if scoring == 'cards' else 1
                    num_rounds += 1
                    break
            # if len(window) == 3:
            #     t = tuple(window)
            #     p1_hit = (t == s1)
            #     p2_hit = (t == s2)

            #     if p1_hit and p2_hit:
            #         # Decide how to handle simultaneous hits. 
            #         # Usually, you split the point or it's a draw for that round.
            #         score1 += 0.5 
            #         score2 += 0.5
            #         num_rounds += 1
            #         break 
            #     elif p1_hit:
            #         score1 += cards if scoring == 'cards' else 1
            #         num_rounds += 1
            #         break
            #     elif p2_hit:
            #         score2 += cards if scoring == 'cards' else 1
            #         num_rounds += 1
            #         break

        if 52 - i < 3:
            break

    winner = 1 if score1 > score2 else 2 if score2 > score1 else 0
    return winner, score1, score2, i, num_rounds

def run_simulation(
        deck_file: str,
        seq1: tuple[str, ...],
        seq2: tuple[str, ...],
        scoring: ScoringMode = 'cards'
    ) -> tuple[float, float, float, float, float, float]:

    '''
    Run Ron's Game for the total number of trials
    
    :param deck_file: Path to all the raw decks already generated
    :type deck_file: str
    :param seq1: Player 1 sequence of colors
    :type seq1: tuple[str, ...]
    :param seq2: Player 2 sequence of colors
    :type seq2: tuple[str, ...]
    :param scoring: cards or rounds
    :type scoring: ScoringMode
    :return: A tuple of all statistics for that matchup
    :rtype: tuple[float, float, float, float, float, float]
    '''

    p1 = p2 = ties = 0
    score_diff_sum = game_len_sum = rounds_sum = 0.0

    data = np.load(deck_file)
    decks = data['decks']

    trials = len(decks)

    # For the total number of trials
    for i in trange(
        trials,
        desc=f'{scoring} | {''.join(seq1)} vs {''.join(seq2)}',
        leave=False
    ):
        # Simluate one game and add the statistics to the tracker variables above
        result, s1, s2, gl, r = simulate_rons_game(seq1, seq2, scoring, deck=decks[i])
        score_diff_sum += (s2 - s1)
        game_len_sum += gl
        rounds_sum += r

        if result == 1:
            p1 += 1
        elif result == 2:
            p2 += 1
        else:
            ties += 1

    # Create an array that averages by the number of trials
    result_array = np.array([
        p1 / trials,
        p2 / trials,
        ties / trials,
        score_diff_sum / trials,
        game_len_sum / trials,
        rounds_sum / trials,
    ])

    # Save the results
    name = f'{scoring}_{''.join(seq1)}_vs_{''.join(seq2)}'
    out_path = os.path.join('data/results', f'{name}.npy')

    np.save(out_path, result_array)
    print(f'Saved results → {out_path}')

    # Turn the array into a tuple
    return tuple(result_array)

def example_game():  # David's contribution
    print("\n=== Penney's Game (Example) ===\n")

    # the player chooses a sequence, only being allowed the parameters of the game
    while True:
        p1_seq = input("Choose a 3-color sequence from a deck of cards, like RRB (R for Red, B for Black): ").strip().upper()
        if len(p1_seq) == 3 and all(c in ('R','B') for c in p1_seq):
            break
        print("Invalid sequence, please try again. Use only R and B with a length of 3.")

    # tuples the player's selection for the check process
    p1_seq_tuple = tuple(p1_seq)
    print(f"P1 (Player) chooses: {''.join(p1_seq_tuple)}\n")

    # automatically creates and tuple-fies the player 2 selection
    # if the computer generates the same selection as the player, it regenerates
    p2_seq_tuple = tuple(np.random.choice(['R', 'B'], size=3))
    while p1_seq_tuple == p2_seq_tuple:
        p2_seq_tuple = tuple(np.random.choice(['R', 'B'], size=3))
    print(f"P2 (Computer) chooses: {''.join(p2_seq_tuple)}\n")

    deck = np.random.randint(0, 2, size=52)
    print("\nA fresh deck of 52 cards has been generated and will be drawn from.")
    print("When cards are drawn, the current sequence will be checked with each player's to score points.")

    # breaks the current combinations into numbers for checking
    s_p1 = tuple(1 if c == 'R' else 0 for c in p1_seq_tuple)
    s_p2 = tuple(1 if c == 'R' else 0 for c in p2_seq_tuple)

    window = []
    print("\nStarting draws...\n")

    p1_score = 0
    p2_score = 0
    
    # this is the checking process by which cards are pulled & scores are tallied.
    for i, card in enumerate(deck, start=1):
        color = 'R' if card == 1 else 'B'
        print(f"Draw #{i}: {color}")

        window.append(card)
        if len(window) > 3:  # always scoring by rounds, so unscored overflow cards must be removed
            window.pop(0)

        if len(window) == 3:
            t = tuple(window)    # tupled for consistency with the player selections.
            print(f"  Current window: {''.join('R' if x else 'B' for x in t)}")

            if t == s_p1:
                print("\n>>> P1 SCORES! Your sequence appeared first.\n")
                p1_score = p1_score + 1
                window.clear()             # the cards that were just scored are cleared from the window
                continue
            
            if t == s_p2:
                print("\n>>> P2 SCORES! The computer's sequence appeared first.\n")
                p2_score = p2_score + 1
                window.clear()             # the cards that were just scored are cleared from the window
                continue

        if p1_score > p2_score:
            winner = "P1 (You)"
        elif p1_score < p2_score:
            winner = "P2 (Computer)"
        else:
            winner = "neither player! It's a tie"

    print(f"\nAnd the winner is... {winner}!\n")
    print(f"P1 Score: {p1_score}")
    print(f"P2 Score: {p2_score}")

    print('\nThis example is finished. Keep in mind that this scoring system is by "rounds". The "--scoring rounds" argument is used.\n')
    print('If you want to score by cards (raw count of cards scored), then use the "--scoring cards" argument which is default, anyway.\n')
    return
