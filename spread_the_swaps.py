import numpy as np
from typing import Dict, Tuple


basic_swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])


def swap(i: int, n_qubits: int = 2) -> np.ndarray:
    # swap qubit i with qubit i+1
    s = basic_swap
    for j in range(i):
        s = np.kron(np.eye(2), s)
    for j in range(n_qubits - i - 2):
        s = np.kron(s, np.eye(2))
    return s


def get_key(n_qubits: int) -> Dict[Tuple[int], np.ndarray]:
    return {(i, i+1): swap(i, n_qubits) for i in range(n_qubits - 1)}


