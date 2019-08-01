import numpy as np
from scipy.linalg import sqrtm
import math
from density_runner import apply_channel, run_by_matrices, ops
from metrics.diamond_distance import random_densities
from typing import Iterable, Any, Dict, Optional


def fidelity(rho1, rho2):
    s = sqrtm(rho1)
    return np.trace(sqrtm(s @ rho2 @ s)).real ** 2


def f_min(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray], n_trials: int) -> float:
    dim = channel1[0].shape[0]
    min_fidelity = min([fidelity(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
                        random_densities(dim, n_trials)])
    for k in range(int(math.log(dim, 2))):
        channel1 = [np.kron(e, np.eye(2)) for e in channel1]
        channel2 = [np.kron(e, np.eye(2)) for e in channel2]
        with_ancilla = min([fidelity(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
                            random_densities(2 ** (k + 1) * dim, n_trials)])
        min_fidelity = min(min_fidelity, with_ancilla)
    return min_fidelity


def experimental(circuit_string: Iterable[Any], unitary: np.ndarray, n_trials: int, noise_channels: Iterable = [],
                 key: Dict[Any, np.ndarray] = None) -> float:
    dim = unitary.shape[0]
    if key is None:
        key = ops
    min_fidelity = min([fidelity(run_by_matrices(circuit_string, xi, noise_channels, key),
                                 apply_channel([unitary], xi)) for xi in random_densities(dim, n_trials)])
    for i in range(int(math.log(dim, 2))):
        key = {k: np.kron(v, np.eye(2)) for k, v in key.items()}
        unitary = np.kron(unitary, np.eye(2))
        noise_channels = [[np.kron(e, np.eye(2)) for e in c]for c in noise_channels]
        with_ancilla = min([fidelity(run_by_matrices(circuit_string, xi, noise_channels, key=key),
                            apply_channel([unitary], xi)) for xi in random_densities(2 ** (i + 1) * dim, n_trials)])
        min_fidelity = min(min_fidelity, with_ancilla)
    return min_fidelity


def effect_of_noise(circuit_description: Iterable[Any], noise_channels: Iterable = [],  n_qubits: int = 1,
                    circuit_key: Optional[Dict[Any, np.ndarray]] = None) -> float:
    d = 2 ** n_qubits
    unitary = np.eye(d)
    for s in circuit_description[::-1]:
        unitary = circuit_key[s] @ unitary
    return 1 - experimental(circuit_description, unitary, 100, noise_channels, key=circuit_key)


def angle(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray]) -> float:
    return math.acos(f_min(channel1, channel2, 100) ** 0.5)


def bures(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray]) -> float:
    return (2 - 2 * f_min(channel1, channel2, 100) ** 0.5) ** 0.5


def C(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray]) -> float:  # That's the only name they give it
    return (1 - f_min(channel1, channel2, 100)) ** 0.5
