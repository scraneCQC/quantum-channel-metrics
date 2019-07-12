import random
import numpy as np
from scipy.linalg import sqrtm
import math
from density_runner import apply_channel, run_by_matrices, run_with_noisy_ancilla


def pure_density_from_state(state):
    return np.outer(state.conjugate(), state)


def fidelity(rho1, rho2):
    s = sqrtm(rho1)
    return np.trace(sqrtm(s @ rho2 @ s)).real ** 2


def random_state(dim):
    # might want to put a different distribution on this, idk
    c = np.array([complex(random.random() - 0.5, random.random() - 0.5) for _ in range(dim)])
    squared_modulus = c @ c.conjugate()
    return c / (squared_modulus ** 0.5)


def f_min(channel1, channel2, n_trials):
    # TODO: optimise over all tensor products up to size n
    dim = channel1[0].shape[0]
    without_ancilla = min([fidelity(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
                           [pure_density_from_state(random_state(dim)) for _ in range(n_trials)]])
    channel1_ancilla = [np.kron(e, np.eye(dim)) for e in channel1]
    channel2_ancilla = [np.kron(e, np.eye(dim)) for e in channel2]
    with_ancilla = min([fidelity(apply_channel(channel1_ancilla, xi), apply_channel(channel2_ancilla, xi)) for xi in
                        [pure_density_from_state(random_state(dim ** 2)) for _ in range(n_trials)]])
    return min(with_ancilla, without_ancilla)


def experimental(circuit_string, unitary, n_trials, *, p1=0, gamma1=0, gamma2=0):
    # TODO: optimise over all tensor products up to size n
    dim = unitary.shape[0]
    without_ancilla = min([fidelity(run_by_matrices(circuit_string, xi, p1, gamma1, gamma2),
                                    apply_channel([unitary], xi)) for xi in
                           [pure_density_from_state(random_state(dim)) for _ in range(n_trials)]])
    with_ancilla = min([fidelity(run_with_noisy_ancilla(circuit_string, xi, p1, gamma1, gamma2),
                                 apply_channel([np.kron(unitary, np.eye(2))], xi)) for xi in
                        [pure_density_from_state(random_state(dim ** 2)) for _ in range(n_trials)]])
    return min(with_ancilla, without_ancilla)


def angle(channel1, channel2):
    return math.acos(f_min(channel1, channel2, 100) ** 0.5)


def bures(channel1, channel2):
    return (2 - 2 * f_min(channel1, channel2, 100) ** 0.5) ** 0.5


def C(channel1, channel2):  # That's the only name they give it
    return (1 - f_min(channel1, channel2, 100)) ** 0.5
