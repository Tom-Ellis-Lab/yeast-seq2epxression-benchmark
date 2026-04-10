import numpy as np

nucleotide2onehot = {
    "A": [1, 0, 0, 0],
    "C": [0, 1, 0, 0],
    "G": [0, 0, 1, 0],
    "T": [0, 0, 0, 1],
    "N": [0, 0, 0, 0],
}


def one_hot_encode(seq: str) -> np.ndarray:
    return np.array([nucleotide2onehot.get(base, [0, 0, 0, 0]) for base in seq])
