import os
import numpy as np

def generate_and_save_decks(
    trials: int,
    seed: int | None = None,
    regen: bool = False,
) -> None:
    path = os.path.join('data/decks', f'decks_{trials}.npz')

    # If path already exists and not regen, raise an error
    if os.path.exists(path) and not regen:
        raise FileExistsError(
            f'{path} already exists. Use --regen to replace it.'
        )

    if seed is not None:
        np.random.seed(seed)

    # Create the decks of random 0 or 1
    decks = np.random.randint(
        0, 2,
        size=(trials, 52),
        dtype=np.uint8
    )

    # Save the generated decks to a compressed npy file
    np.savez_compressed(
        path,
        decks=decks,
        trials=trials,
        seed=seed,
    )

    print(f'Saved {trials} decks → {path}')
    return

def generate_decks(trials: int):
    base_deck = np.array([0]*26 + [1]*26, dtype=np.uint8)
    
    # 2. Initialize an empty array to hold all decks
    decks = np.empty((trials, 52), dtype=np.uint8)
    
    # 3. Shuffle a fresh copy for every trial
    for i in range(trials):
        # We use a copy so we don't exhaust the base_deck
        shuffled = base_deck.copy()
        np.random.shuffle(shuffled)
        decks[i] = shuffled
        
    return decks