# Rons' Game
A version of [Penney's Game](https://en.wikipedia.org/wiki/Penney%27s_game) played with a deck of cards!

Two players each select a sequence of three card colors. For example Player 1 selects "BBB" and Player 2 selects "RBR". Then cards are flipped from the top of the deck until a sequence of three cards matches one of the chosen color sequences. This continues for the entire deck. If the last card is flipped over and no sequence of three colors is made, the round does not score.

The scoring of the original Penney's Game is done by counting the number of "rounds" won within each game of coinflips. This can be replicated in the card version. However, the game morphs when instead of counting individual rounds, the score is based on the number of cards won in the sequence of cards flipped and then added to the previous score for that player. For example,

Player 1 (BBB) - Player 2 (RBR)     
Card Sequence: "BRRBB<u>RBR</u>" -> Results in a Player 2 winning with a score of 8 being added to the overall score.

---

# Project Motivation
This project simulates all possible combinations of sequences of three card colors for each scoring options. It then produces heatmaps to show the percent chance that Player 2 wins the game.

---

# Example Run in CMD
``` 
uv sync
uv run python -m main --trials 100000 --scoring rounds

# Example flags for running:
--trials -> Number of trials per matchup (default: 100000)
--scoring -> Scoring method: cards or rounds
--regen -> Force regeneration of simulation data for given number of trials, even if cached in data folder
--no-plots -> Do not display plots after simulating games
```

# TODO
- Be able to add 10 or 50 decks to the simulation
- Fix dominance graph plotting
- Add comments for everything
