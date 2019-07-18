import numpy as np
from scipy.linalg import sqrtm
from density_runner import apply_channel, ops
from typing import Iterable, Any, Dict, Optional
from noise import depolarising_channel, amplitude_damping_channel, phase_damping_channel


def trace_norm(m1: np.ndarray, m2: np.ndarray) -> float:
    diff = complex(1, 0) * (m1 - m2)
    s = sqrtm(diff @ diff.transpose().conjugate())
    return np.trace(s).real / 2


def jamiolkowski(channel: Iterable[np.ndarray]) -> np.ndarray:
    d = channel[0].shape[0]
    big_channel = [np.kron(e, np.eye(d)) for e in channel]
    phi = sum([np.kron(c, c) for c in np.eye(d)])
    start_rho = np.outer(phi, phi) / d
    return apply_channel(big_channel, start_rho)


def j_distance(channel1: Iterable[np.ndarray], channel2: Iterable[np.ndarray]) -> float:
    return trace_norm(jamiolkowski(channel1), (jamiolkowski(channel2)))


def circuit_to_jamiolkowski(circuit_description: Iterable[Any], circuit_key: Optional[Dict[Any, np.ndarray]] = None, *,
                            n_qubits: int = 1, noise_channels: Iterable = []) -> np.ndarray:
    if circuit_key is None:
        circuit_key = ops
    d = 2 ** n_qubits

    phi = sum([np.kron(c, c) for c in np.eye(d)])
    rho = np.outer(phi, phi) / d
    for j in circuit_description[::-1]:
        u = circuit_key[j]
        rho = apply_channel([np.kron(u, np.eye(d))], rho)
        for c in noise_channels:
            rho = apply_channel([np.kron(e, np.eye(d)) for e in c], rho)
    return rho


def j_distance_experimental(circuit_description: Iterable[Any], unitary: np.ndarray, noise_channels,
                            circuit_key: Optional[Dict[Any, np.ndarray]] = None, *,
                            n_qubits: int = 1) -> float:
    return trace_norm(circuit_to_jamiolkowski(circuit_description, circuit_key, n_qubits=n_qubits, noise_channels=noise_channels),
                      jamiolkowski([unitary]))


def effect_of_noise(circuit_description: Iterable[Any], noise_channels: Iterable = [], *,
                    circuit_key: Optional[Dict[Any, np.ndarray]] = None, n_qubits: int = 1) -> float:
    return trace_norm(circuit_to_jamiolkowski(circuit_description, circuit_key, n_qubits=n_qubits, noise_channels=noise_channels),
                      circuit_to_jamiolkowski(circuit_description, circuit_key, n_qubits=n_qubits))

# "so far as we are aware, experimentally determining D_pro requires doing full process
# tomography, which for a d-dimensional quantum system
# requires the estimation of d ** 4 − d ** 2 observable averages"

