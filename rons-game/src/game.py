import numpy as np
import os
import numpy as np
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

    # s1 = tuple(1 if c == 'R' else 0 for c in seq1)
    # s2 = tuple(1 if c == 'R' else 0 for c in seq2)

    # score1 = score2 = 0
    # i = 0

    # while i <= 49:
    #     # We look for a match starting from current position 'i'
    #     found_round_winner = False
        
    #     # We must deal at least 3 cards to check a window
    #     for j in range(i + 2, 52):
    #         # Check the window ending at card 'j'
    #         window = tuple(deck[j-2 : j+1])
            
    #         p1_hit = (window == s1)
    #         p2_hit = (window == s2)
            
    #         if p1_hit or p2_hit:
    #             if p1_hit and p2_hit:
    #                 score1 += 1
    #                 score2 += 1
    #             elif p1_hit:
    #                 score1 += 1
    #             elif p2_hit:
    #                 score2 += 1
                
    #             # DISCARD RULE: Move 'i' to the card AFTER this win
    #             i = j + 1
    #             found_round_winner = True
    #             break
        
    #     if not found_round_winner:
    #         break # No more cards left to form a 3-card sequence

    # # WINNER CALCULATION
    # if score1 > score2:
    #     return 1, int(score1), int(score2), 52, 0
    # if score2 > score1:
    #     return 2, int(score1), int(score2), 52, 0
    # return 0, int(score1), int(score2), 52, 0 # TIE (Draw)

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
