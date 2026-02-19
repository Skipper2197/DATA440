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
