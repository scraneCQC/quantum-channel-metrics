import random
import numpy as np
from scipy.linalg import sqrtm
import math
from density_runner import apply_channel, run_by_matrices, ops


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


def random_densities(dim, n_trials):
    return [sum([r * pure_density_from_state(random_state(dim)) for r in np.random.dirichlet(np.ones(dim), size=1)[0]])
            for _ in range(n_trials)]


def f_min(channel1, channel2, n_trials):
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


def experimental(circuit_string, unitary, n_trials, *, p1=0, gamma1=0, gamma2=0, key=None):
    dim = unitary.shape[0]
    if key is None:
        key = ops
    min_fidelity = min([fidelity(run_by_matrices(circuit_string, xi, p1, gamma1, gamma2, key),
                                 apply_channel([unitary], xi)) for xi in random_densities(dim, n_trials)])
    for i in range(int(math.log(dim, 2))):
        key = {k: np.kron(v, np.eye(2)) for k, v in key.items()}
        unitary = np.kron(unitary, np.eye(2))
        with_ancilla = min([fidelity(run_by_matrices(circuit_string, xi, p1, gamma1, gamma2, key=key),
                            apply_channel([unitary], xi)) for xi in random_densities(2 ** (i + 1) * dim, n_trials)])
        min_fidelity = min(min_fidelity, with_ancilla)
    return min_fidelity


def angle(channel1, channel2):
    return math.acos(f_min(channel1, channel2, 100) ** 0.5)


def bures(channel1, channel2):
    return (2 - 2 * f_min(channel1, channel2, 100) ** 0.5) ** 0.5


def C(channel1, channel2):  # That's the only name they give it
    return (1 - f_min(channel1, channel2, 100)) ** 0.5
