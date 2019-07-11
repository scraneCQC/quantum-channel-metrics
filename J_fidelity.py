from Pauli import *
import math
from approximation_runner import get_expectation
import density_runner


def apply_channel(channel, density):
    return sum([e @ density @ e.transpose().conjugate() for e in channel])


def f_pro(channel, unitary):
    dim = channel[0].shape[0]
    n_qubits = int(math.log(dim, 2))  # please don't give me qudits, the Pauli's aren't nice
    u_basis = get_diracs(n_qubits)
    basis_one = [I1, I1 + X, I1 + Y, I1 + Z]
    state_basis = basis_one
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in basis_one]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    return 1 / dim ** 3 * \
        sum([np.trace(sigmas[k] @ apply_channel(channel, state_basis[k])) for k in range(dim ** 2)]).real
    # Experimentally, this trace is the expectation value of sigma.
    # There are d ** 2 observables whose expectations would need to be found.
    # If we cannot measure these we need to use a different formula which requires up to d ** 4 - d ** 2 expectations


def f_pro_experimental(circuit_string, unitary, p1=0, gamma1=0, gamma2=0):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    u_basis = get_diracs(n_qubits)
    basis_one = [I1, I1 + X, I1 + Y, I1 + Z]
    state_basis = basis_one
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in basis_one]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    return 1 / dim ** 3 * \
        sum([np.trace(sigmas[k] @ density_runner.run_by_matrices(circuit_string, state_basis[k], p1, gamma1, gamma2))
            for k in range(dim ** 2)]).real

    # This one would run the circuit with a noise simulation on AerBackend and get expectation by repeated measurement:
    # But I can't get the expectation bit working
    return sum([get_expectation(sigmas[k], circuit_string, p1, gamma1, gamma2) for k in range(dim ** 2)]).real


def angle(channel, unitary):
    return math.acos(f_pro(channel, unitary) ** 0.5)


def bures(channel, unitary):
    return (2 - 2 * f_pro(channel, unitary) ** 0.5) ** 0.5


def C(channel, unitary):  # That's the only name they give it
    return (1 - f_pro(channel, unitary)) ** 0.5