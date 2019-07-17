import numpy as np
from scipy.linalg import sqrtm, block_diag
import random
from Pauli import *
from density_runner import apply_channel, run_by_matrices, ops
import math
from typing import List, Iterable, Any, Dict, Callable


def trace_norm(m1: np.ndarray, m2: np.ndarray) -> float:
    diff = complex(1, 0) * (m1 - m2)
    return np.trace(sqrtm(diff @ diff.transpose().conjugate())).real / 2


def random_state(dim: int) -> np.ndarray:
    # might want to put a different distribution on this, idk
    c = np.array([complex(random.random() - 0.5, random.random() - 0.5) for _ in range(dim)])
    squared_modulus = c @ c.conjugate()
    return c / (squared_modulus ** 0.5)


def density_matrix_to_fano(rho: np.ndarray):
    # TODO: work for higher dimensions than 2
    return [np.trace(rho @ dirac) for dirac in one_qubit_diracs]


def pure_density_from_state(state: np.ndarray) -> np.ndarray:
    return np.outer(state.conjugate(), state)


def random_densities(dim: int, n_trials: int) -> List[np.ndarray]:
    return [sum([r * pure_density_from_state(random_state(dim)) for r in np.random.dirichlet(np.ones(dim), size=1)[0]])
            for _ in range(n_trials)]


def monte_carlo_f_algorithm(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray], n_trials: int) -> float:
    # channels: a list of numpy arrays (Kraus matrices)
    dim = channel1[0].shape[0]
    max_norm = max([trace_norm(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
                    random_densities(dim, n_trials)])
    for k in range(int(math.log(dim, 2))):
        channel1 = [np.kron(e, np.eye(2)) for e in channel1]
        channel2 = [np.kron(e, np.eye(2)) for e in channel2]
        with_ancilla = max([trace_norm(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
                            random_densities(2 ** (k + 1) * dim, n_trials)])
        max_norm = max(max_norm, with_ancilla)
    return max_norm


def monte_carlo_from_circuit(circuit_string: Iterable[Any], unitary: np.ndarray, n_trials: int,
                             p1: float = 0, gamma1: float = 0, gamma2: float = 0,
                             circuit_key: Dict[Any, Callable] = None) -> float:
    if circuit_key is None:
        circuit_key = ops
    dim = unitary.shape[0]
    max_norm = max([trace_norm(run_by_matrices(circuit_string, xi, p1, gamma1, gamma2, circuit_key), apply_channel([unitary], xi))
                    for xi in random_densities(dim, n_trials)])
    for k in range(int(math.log(dim, 2))):
        circuit_key = {i: np.kron(v, np.eye(2)) for i, v in circuit_key.items()}
        unitary = np.kron(unitary, np.eye(2))
        with_ancilla = max([trace_norm(run_by_matrices(circuit_string, xi, p1, gamma1, gamma2, circuit_key),
                            apply_channel([unitary], xi)) for xi in random_densities(2 ** (k + 1) * dim, n_trials)])
        max_norm = max(max_norm, with_ancilla)
    return max_norm
