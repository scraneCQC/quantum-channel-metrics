import numpy as np
from scipy.linalg import sqrtm
import random
import math
from Pauli import *


def trace_norm(m1, m2):
    diff = complex(1, 0) * (m1 - m2)
    return np.trace(sqrtm(diff @ diff.transpose().conjugate())).real / 2


def random_state(dim):
    # might want to put a different distribution on this, idk
    c = np.array([complex(random.random() - 0.5, random.random() - 0.5) for _ in range(dim)])
    squared_modulus = c @ c.conjugate()
    return c / (squared_modulus ** 0.5)


def density_matrix_to_fano(rho):
    # TODO: work for higher dimensions than 2
    return [np.trace(rho @ dirac) for dirac in one_qubit_diracs]


def pure_density_from_state(state):
    return np.outer(state.conjugate(), state)


def apply_channel(channel, density):
    return sum([e @ density @ e.transpose().conjugate() for e in channel])


def monte_carlo_f_algorithm(channel1, channel2, n_trials):
    # channels: a list of numpy arrays (Kraus matrices)
    # TODO: maximise over all tensor products up to size n
    dim = channel1[0].shape[0]
    without_ancilla = max([trace_norm(apply_channel(channel1, xi), apply_channel(channel2, xi)) for xi in
        [pure_density_from_state(random_state(dim)) for _ in range(n_trials)]])
    channel1_ancilla = [np.kron(e, np.eye(dim)) for e in channel1]
    channel2_ancilla = [np.kron(e, np.eye(dim)) for e in channel2]
    with_ancilla = max([trace_norm(apply_channel(channel1_ancilla, xi), apply_channel(channel2_ancilla, xi)) for xi in
        [pure_density_from_state(random_state(dim ** 2)) for _ in range(n_trials)]])
    return max(with_ancilla, without_ancilla)

