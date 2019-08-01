from pytket.backends.ibm import IBMQBackend
from pytket import Circuit
from typing import Iterable, Dict, Any, Callable
from metrics.approximation_runner import circuit_from_string
from itertools import product
import math
from Pauli import get_diracs, I1, X, Y, Z
import numpy as np
from common_gates import Rz

backend = IBMQBackend("ibmq_5_tenerife")

final_circuits = {a: Circuit(1) for a in "IXYZ"}
final_circuits["X"].H(0)
final_circuits["Y"].Sdg(0)
final_circuits["Y"].H(0)


def f_pro_ibmq(circuit_string: Iterable[Any], unitary: np.ndarray, key: Dict[Any, np.ndarray] = None) -> float:
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    u_basis = get_diracs(n_qubits)
    one_qubit_state_basis = [I1 - Z, I1 + X, I1 + Y, I1 + Z]
    state_basis = one_qubit_state_basis
    one_qubit_a = np.array([[1, 0, 0, 1], [-1, 2, 0, -1], [-1, 0, 2, -1], [-1, 0, 0, 1]])
    a = one_qubit_a
    for _ in range(n_qubits - 1):
        state_basis = [np.kron(x, y) for x in state_basis for y in one_qubit_state_basis]
        a = np.kron(a, one_qubit_a)
    sigmas = get_diracs(n_qubits)
    b = np.array([[np.trace(unitary @ u_basis[j] @ unitary.transpose().conjugate() @ sigmas[l]) / dim for l in range(dim ** 2)] for j in range(dim ** 2)])
    m = a.transpose() @ b

    one_qubit_preps = [(lambda i, c: c.X(i)), (lambda i, c: c.H(i)), (lambda i, c: c.Rx(i, -0.5)), (lambda i, c: None)]

    def prepare_state(single_qubit_states):
        circuit = Circuit(n_qubits)
        for i in range(len(single_qubit_states)):
            one_qubit_preps[single_qubit_states[i]](i, circuit)
        return circuit

    preps = [prepare_state(t) for t in product([0, 1, 2, 3], repeat=n_qubits)]
    expectations = np.zeros((dim ** 2, dim ** 2))
    for k, l in list(product(range(dim ** 2), range(dim ** 2))):
        s = list(product("IXYZ", repeat=n_qubits))
        expectations[l][k] = get_pauli_expectation_ibmq(circuit_string, preps[k], "".join(s[l]), key=key)
    return 1 / dim ** 3 * sum(expectations[k][l] * m[l][k] for l in range(dim ** 2) for k in range(dim ** 2)).real


def get_pauli_expectation_ibmq(circuit_string: Iterable[Any], initial_circuit: Circuit, pauli_string: str, *,
                             shots: int = 4096, key: Dict[Any, Callable] = None) -> float:
    n_qubits = len(pauli_string)
    if pauli_string == "I" * n_qubits:
        return 1
    circuit = initial_circuit.copy()
    circuit.add_circuit(circuit_from_string(circuit_string, n_qubits=n_qubits, key=key), list(range(n_qubits)))
    for i in range(n_qubits):
        circuit.add_circuit(final_circuits[pauli_string[i]].copy(), [i])
    circuit.measure_all()
    noisy_shots = backend.get_counts(circuit, shots)
    return sum(v * (-1) ** sum(k) for k, v in noisy_shots.items())/shots


print(f_pro_ibmq("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT", Rz(math.pi / 128, 0, 1)))
