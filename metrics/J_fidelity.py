from Pauli import *
import math
from metrics.approximation_runner import get_pauli_expectation
from metrics import density_runner
from metrics.density_runner import apply_channel
from pytket import Circuit
from itertools import product
from typing import Iterable, Any, Dict, Optional
from pytket.backends import Backend


one_qubit_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]


def f_pro(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    dim = channel[0].shape[0]
    n_qubits = int(math.log(dim, 2))  # please don't give me qudits, the Pauli's aren't nice
    u_basis = get_diracs(n_qubits)
    state_basis = one_qubit_state_basis
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    return 1 / dim ** 3 * \
        sum([np.trace(sigmas[k] @ apply_channel(channel, state_basis[k])) for k in range(dim ** 2)]).real


def f_pro_experimental(circuit_string: Iterable[Any], unitary: np.ndarray, noise_channels: Iterable = [],
                       key: Dict[Any, np.ndarray] = None) -> float:
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    u_basis = get_diracs(n_qubits)
    state_basis = one_qubit_state_basis
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
    a = np.array(np.eye(dim ** 2) -
                 np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
    sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                   for k in range(dim ** 2)]) for l in range(dim ** 2)]
    expectations = [np.trace(sigmas[k] @ density_runner.run_by_matrices(circuit_string, state_basis[k], noise_channels, key))
                    .real for k in range(dim ** 2)]
    # print(expectations)
    return 1 / dim ** 3 * sum(expectations)


dim = 8
n_qubits = 3
u_basis = np.array(get_diracs(n_qubits))
state_basis = [(I1 - Z) / 2, (I1 + X) / 2, (I1 + Y) / 2, (I1 + Z) / 2]
one_qubit_states = state_basis
one_qubit_a = np.array([[1, 0, 0, 1], [-1, 2, 0, -1], [-1, 0, 2, -1], [-1, 0, 0, 1]])
a = one_qubit_a
for _ in range(n_qubits - 1):
    state_basis = [np.kron(x, y) for x in one_qubit_states for y in state_basis]
    a = np.kron(a, one_qubit_a)
state_basis = np.array(state_basis)
sigmas = u_basis
s = list(product("IXYZ", repeat=n_qubits))


one_qubit_preps = [(lambda i, c: c.X(i)), (lambda i, c: c.H(i)), (lambda i, c: c.Rx(i, -0.5)), (lambda i, c: None)]


def prepare_state(single_qubit_states):
    circuit = Circuit(n_qubits)
    for i in range(len(single_qubit_states)):
        one_qubit_preps[single_qubit_states[i]](i, circuit)
    return circuit


preps = [prepare_state(t) for t in product([0, 1, 2, 3], repeat=n_qubits)]


def load_expectations_from_file(filename):
    expectations = np.full((dim ** 2, dim ** 2), np.nan)
    with open(filename) as file:
        lines = file.readlines()
    for line in lines:
        d = line.split()
        expectations[int(d[0])][int(d[1])] = float(d[2])
    return expectations


def save_expectations_to_file(filename, expectations: np.ndarray):
    with open(filename, "w") as file:
        file.writelines([str(l) + " " + str(k) + " " + str(expectations[l][k]) + " " + "".join(s[l]) + "\n"
                         for l in range(dim ** 2) for k in range(dim ** 2) if not np.isnan(expectations[l][k])])


def f_pro_simulated(circuit: Circuit, unitary: np.ndarray, backend: Backend, filename) -> float:
    b = np.einsum('ab,jbc,dc,lda->lj', unitary, u_basis, unitary.conjugate(), sigmas, optimize=True) / dim
    m = b @ a
    expectations = load_expectations_from_file(filename)
    for k, l in list(product(range(dim ** 2), range(dim ** 2))):
        if np.isnan(expectations[l][k]):
            if abs(m[l][k]) > 1e-5:
                expectations[l][k] = get_pauli_expectation(circuit, preps[k], "".join(s[l]), backend, shots=8192)
        if k == l:
            save_expectations_to_file(filename, expectations)
    expectations = np.where(np.isnan(expectations), 0, expectations)
    return 1 / dim ** 3 * np.einsum('lk,lk->', expectations, m).real


def effect_of_noise(circuit_description: Iterable[Any], noise_channels: Iterable = [],  n_qubits: int = 1,
                    circuit_key: Optional[Dict[Any, np.ndarray]] = None) -> float:
    d = 2 ** n_qubits
    unitary = np.eye(d)
    for s in circuit_description[::-1]:
        unitary = circuit_key[s] @ unitary
    return 1 - f_pro_experimental(circuit_description, unitary, noise_channels, key=circuit_key)


def angle(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    return math.acos(f_pro(channel, unitary) ** 0.5)


def bures(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
    f = max(f_pro(channel, unitary), 0)
    return (1 - f ** 0.5) ** 0.5


def C(channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:  # That's the only name they give it
    return (1 - f_pro(channel, unitary)) ** 0.5
