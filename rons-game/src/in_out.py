import os
import numpy as np

# -------------------
# Scoring Methods
# -------------------

def heatmap_file(trials: int, scoring: str) -> str:
    '''
    Get file path of the scored decks for the number fo specified trials
    '''
    return os.path.join('data/results', f'{scoring}_{trials}.npz')


def save_data(data: dict, trials: int, scoring: str) -> None:
    '''
    Save the scored matchup results to a compressed np file
    '''
    seq1 = []
    seq2 = []
    values = []

    for (s1, s2), v in data.items():
        seq1.append(s1)
        seq2.append(s2)
        values.append(v)

    np.savez_compressed(
        heatmap_file(trials, scoring),
        seq1=np.array(seq1, dtype='U3'),
        seq2=np.array(seq2, dtype='U3'),
        values=np.array(values, dtype=np.float64),
    )

    print(f'Saved matchup results → {heatmap_file(trials, scoring)}')


def load_data(trials: int, scoring: str) -> dict:
    '''
    Load the scored matchups and extract to a python dictionary
    '''
    path = heatmap_file(trials, scoring)
    if not os.path.exists(path):
        raise FileNotFoundError(f'No saved data found at {path}')

    npz = np.load(path)

    seq1 = npz['seq1']
    seq2 = npz['seq2']
    values = npz['values']

    data = {}
    for i in range(len(seq1)):
        data[(seq1[i], seq2[i])] = tuple(values[i])
    return data


def data_exists(trials: int, scoring: str) -> bool:
    '''
    Check if the scored matchup file exists for the specified number of trials
    '''
    return os.path.exists(heatmap_file(trials, scoring))


# -------------------
# Raw Deck Methods
# -------------------

def decks_file(trials: int) -> str:
    '''
    Get file path of the raw decks for the number fo specified trials
    '''
    return os.path.join('data/decks', f'decks_{trials}.npz')


def save_decks(decks: np.ndarray, trials: int) -> None:
    '''
    Save decks of the specified number of trials to compressed np file
    '''
    np.savez_compressed(decks_file(trials), decks=decks)
    print(f'Saved {len(decks)} decks → {decks_file(trials)}')


def load_decks(trials: int) -> np.ndarray:
    '''
    Load raw decks of the specified trial number.
    If that file does not exist, raise an error, becuase it SHOULD!
    '''
    path = decks_file(trials)
    if not os.path.exists(path):
        raise FileNotFoundError(f'No decks file found at {path}')
    return np.load(path)['decks']


def decks_exists(trials: int) -> bool:
    '''
    Check if the raw deck file exists for the specified number of trials
    '''
    return os.path.exists(decks_file(trials))
