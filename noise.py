from Pauli import *
from typing import List


def not_channel(n_qubits: int) -> List[np.ndarray]:
    u = X
    for _ in range(n_qubits - 1):
        u = np.kron(u, X)
    return [u]


def identity_channel(n_qubits: int) -> List[np.ndarray]:
    return [np.eye(2 ** n_qubits)]


def depolarising_channel(p1: float, n_qubits: int = 1) -> List[np.ndarray]:
    basic_channel = [p * M for p, M in zip(
            [(1 - p1) ** 0.5, (p1 / 3) ** 0.5, (p1 / 3) ** 0.5, (p1 / 3) ** 0.5], one_qubit_diracs)]
    channel = basic_channel
    for _ in range(n_qubits - 1):
        channel = [np.kron(channel[i], basic_channel[i]) for i in range(4)]
    return channel


def amplitude_damping_channel(gamma: float, n_qubits: int = 1) -> List[np.ndarray]:
    basic_channel = [np.array([[1, 0], [0, (1 - gamma) ** 0.5]]), np.array([[0, gamma ** 0.5], [0, 0]])]
    channel = basic_channel
    for _ in range(n_qubits - 1):
        channel = [np.kron(channel[i], basic_channel[i]) for i in range(2)]
    return channel


def phase_damping_channel(gamma: float, n_qubits: int = 1) -> List[np.ndarray]:
    basic_channel = [np.array([[1, 0], [0, (1 - gamma) ** 0.5]]), np.array([[0, 0], [0, gamma ** 0.5]])]
    channel = basic_channel
    for _ in range(n_qubits - 1):
        channel = [np.kron(channel[i], basic_channel[i]) for i in range(2)]
    return channel