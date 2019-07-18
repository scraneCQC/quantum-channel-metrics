from Pauli import *
import math
from approximation_runner import get_pauli_expectation_v2
import density_runner
from density_runner import apply_channel
from pytket import Circuit
from itertools import product
import time
from typing import Iterable, Any, Dict, Optional


def f_pro(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    dim = channel[0].shape[0]
    n_qubits = int(math.log(dim, 2))  # please don't give me qudits, the Pauli's aren't nice
    u_basis = get_diracs(n_qubits)
    one_qubit_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]
    state_basis = one_qubit_state_basis
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    return 1 / dim ** 3 * \
        sum([np.trace(sigmas[k] @ apply_channel(channel, state_basis[k])) for k in range(dim ** 2)]).real


def f_pro_experimental(circuit_string: Iterable[Any], unitary: np.ndarray, p1: float = 0, gamma1: float = 0,
                       gamma2: float = 0, key: Dict[Any, np.ndarray] = None) -> float:
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    u_basis = get_diracs(n_qubits)
    one_qubit_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]
    state_basis = one_qubit_state_basis
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    expectations = [np.trace(sigmas[k] @ density_runner.run_by_matrices(circuit_string, state_basis[k], p1, gamma1, gamma2, key))
                    .real for k in range(dim ** 2)]
    # print(expectations)
    return 1 / dim ** 3 * sum(expectations)


def f_pro_simulated(circuit_string: Iterable[Any], unitary: np.ndarray, p1: float = 0, gamma1: float = 0,
                       gamma2: float = 0, key: Dict[Any, np.ndarray] = None) -> float:
    print(circuit_string)
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    u_basis = get_diracs(n_qubits)
    one_qubit_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]
    state_basis = one_qubit_state_basis
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = get_diracs(n_qubits)
    b = np.array([[np.trace(unitary @ u_basis[j] @ unitary.transpose().conjugate() @ sigmas[l]) / dim for l in range(dim ** 2)] for j in range(dim ** 2)])
    m = a.transpose() @ b

    def make_unknown(i, c):
        c.H(i)
        c.Measure(i)
    one_qubit_preps = [make_unknown, (lambda i, c: c.H(i)), (lambda i, c: c.Rx(i, -0.5)), (lambda i, c: None)]

    def prepare_state(single_qubit_states):
        # single_qubit_state: a list of integers where {0: I, 1: I+X, 2: I+Y, 3: I+Z} for density of i'th qubit
        circuit = Circuit(n_qubits)
        for i in range(len(single_qubit_states)):
            one_qubit_preps[single_qubit_states[i]](i, circuit)
        return circuit

    preps = [prepare_state(t) for t in product([0, 1, 2, 3], repeat=n_qubits)]

    # expectations = np.array([[np.trace(sigmas[l] @ density_runner.run_by_matrices(circuit_string, state_basis[k], p1, gamma1, gamma2, key))
    #                .real for k in range(dim ** 2)] for l in range(dim ** 2)])

    expectations = np.zeros((dim ** 2, dim ** 2))
    for k, l in list(product(range(dim ** 2), range(dim ** 2))):
        s = list(product("IXYZ", repeat=n_qubits))
        expectations[l][k] = get_pauli_expectation_v2(circuit_string, preps[k], "".join(s[l]),
                                                   p1=p1, gamma1=gamma1, gamma2=gamma2, key=key, shots=int(1e5)) * dim
    #expectations = np.array([[get_pauli_expectation(circuit_string, prep, "".join(s), n_qubits,
    #                                                p1=p1, gamma1=gamma1, gamma2=gamma2, key=key, shots=100000) * dim
    #                          for prep in preps] for s in product("IXYZ", repeat=n_qubits)])

    return 1 / dim ** 3 * sum(expectations[k][l] * m[l][k] for l in range(dim ** 2) for k in range(dim ** 2)).real


def effect_of_noise(circuit_description: Iterable[Any], p1: float, gamma1: float, gamma2: float,  n_qubits: int = 1,
                    circuit_key: Optional[Dict[Any, np.ndarray]] = None) -> float:
    d = 2 ** n_qubits
    unitary = np.eye(d)
    for s in circuit_description[::-1]:
        unitary = circuit_key[s] @ unitary
    return 1 - f_pro_experimental(circuit_description, unitary, p1=p1, gamma1=gamma1, gamma2=gamma2, key=circuit_key)


def angle(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    return math.acos(f_pro(channel, unitary) ** 0.5)


def bures(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    return (2 - 2 * f_pro(channel, unitary) ** 0.5) ** 0.5


def C(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float: # That's the only name they give it
    return (1 - f_pro(channel, unitary)) ** 0.5
