import numpy as np
from typing import List


I1 = np.array([[1, 0], [0, 1]])
X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]])
Z = np.array([[1, 0], [0, -1]])
one_qubit_diracs = [I1, X, Y, Z]


def get_diracs(n_qubits: int) -> List[np.ndarray]:
    diracs = one_qubit_diracs
    while n_qubits > 1:
        diracs = [np.kron(x, y) for x in diracs for y in one_qubit_diracs]
        n_qubits = n_qubits - 1
    return diracs
