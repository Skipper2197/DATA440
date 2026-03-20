# Rons' Game
A version of [Penney's Game](https://en.wikipedia.org/wiki/Penney%27s_game) played with a deck of cards!

Two players each select a sequence of three card colors. For example, Player 1 selects "BBB" and Player 2 selects "RBR". Then, the cards are flipped from the top of the deck until a sequence of three cards matches one of the chosen color sequences. This continues for the entire deck. If the last card is flipped over and no sequence of three colors is made, the round does not score.

The scoring of the original Penney's Game is done by counting the number of "rounds" won within each game of coin flips. This can be replicated in the card version. However, the game morphs when, instead of counting individual rounds, the score is based on the number of cards won in the sequence of cards flipped and then added to the previous score for that player. For example,

Player 1 (BBB) - Player 2 (RBR)     
Card Sequence: "BRRBB<u>RBR</u>" -> Results in Player 2 winning with a score of 8 being added to the overall score.

``` 
# To enhance your understanding of the game, run the following code:
uv run python -m main --example_game
```

---

# Project Motivation
This project simulates all possible combinations of sequences of three card colors for each scoring option. It then produces heatmaps to show the percent chance that Player 2 wins the game. This way, it's possible to infer the best strategies and counterstrategies for playing the game from a theoretical standpoint.

---

# Example Run in CMD
``` 
uv sync
uv run python -m main --trials 100000 --scoring rounds

# Example flags for running:
--trials -> Number of trials per matchup (default: 100000)
--scoring -> Scoring method: cards or rounds
--regen -> Force regeneration of scored decks, even if cached in data/results folder
--no-plots -> Do not display plots after simulating games
--debug -> Print debug points during run (only intended for debugging)
--example_game -> Run through one simulation of the game with user input to understand the game
```

# Our Findings
Our results are consistent with the original findings of Penney's Game. That is, given Player 1's sequence, flip the middle card (R->B or B->R) and place it at the beginning of the sequence. For example, when Player 1 chooses RBB, Player 2 should choose RRB to maximize their chances of winning.     

These results are true regardless of the version of the game you play (rounds vs cards). Playing the cards scoring version, Player 2 has an increased chance of winning due to the fact that the probability of tying in this version, compared to rounds scoring, is much lower.
