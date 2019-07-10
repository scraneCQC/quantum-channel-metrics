from Pauli import *

not_channel = [one_qubit_diracs[1]]


def identity_channel(n_qubits):
    return [np.eye(2 ** n_qubits)]


def depolarising_channel(p1):
    return [p * M for p, M in zip(
            [(1 - p1) ** 0.5, (p1 / 3) ** 0.5, (p1 / 3) ** 0.5, (p1 / 3) ** 0.5], one_qubit_diracs)]


def amplitude_damping_channel(gamma):
    return [np.array([[1, 0], [0, (1 - gamma) ** 0.5]]), np.array([[0, gamma ** 0.5], [0, 0]])]


def phase_damping_channel(gamma):
    return [np.array([[1, 0], [0, (1 - gamma) ** 0.5]]), np.array([[0, 0], [0, gamma ** 0.5]])]