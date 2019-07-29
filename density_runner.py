from noise import *
from typing import Iterable, Any, Dict, Callable


def apply_channel(channel: Iterable[np.ndarray], density: np.ndarray) -> np.ndarray:
    return sum([e @ density @ e.transpose().conjugate() for e in channel])


ops = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
       "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
       "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
       "X": np.array([[0, 1], [1, 0]]),
       "W": complex(0.5 ** 0.5, 0.5 ** 0.5) * np.eye(2),
       "I": np.eye(2)}


def run_by_matrices(string: Any, start_density: np.ndarray, noise_channels: Iterable,
                    key: Dict[Any, Callable] = None):
    if key is None:
        key = ops
    rho = start_density
    for s in string[::-1]:
        rho = key[s] @ rho @ key[s].transpose().conjugate()
        for c in noise_channels:
            rho = sum(e @ rho @ e.transpose().conjugate() for e in c)
    return rho
