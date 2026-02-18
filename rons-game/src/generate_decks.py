import os
import numpy as np

def generate_and_save_decks(
    trials: int,
    seed: int | None = None,
    overwrite: bool = False,
):
    path = os.path.join('data/decks', f"decks_{trials}.npz")

    if os.path.exists(path) and not overwrite:
        raise FileExistsError(
            f"{path} already exists. Use overwrite=True to replace it."
        )

    if seed is not None:
        np.random.seed(seed)

    decks = np.random.randint(
        0, 2,
        size=(trials, 52),
        dtype=np.uint8
    )

    np.savez_compressed(
        path,
        decks=decks,
        trials=trials,
        seed=seed,
    )

    print(f"Saved {trials} decks → {path}")
