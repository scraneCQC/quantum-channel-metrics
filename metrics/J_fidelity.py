from Pauli import *
import math
from metrics.approximation_runner import get_pauli_expectation
from metrics import density_runner
from metrics.density_runner import apply_channel
from pytket import Circuit
from itertools import product
from typing import Iterable, Any, Dict, Optional
from pytket.backends import Backend
from openfermion import QubitOperator


class ProcessFidelityFinder():
    simple_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]
    pure_state_basis = [(I1 - Z) / 2, (I1 + X) / 2, (I1 + Y) / 2, (I1 + Z) / 2]
    one_qubit_a = np.array([[1, 0, 0, 1], [-1, 2, 0, -1], [-1, 0, 2, -1], [-1, 0, 0, 1]])
    one_qubit_preps = [(lambda i, c: c.X(i)), (lambda i, c: c.H(i)), (lambda i, c: c.Rx(i, -0.5)), (lambda i, c: None)]

    def __init__(self, n_qubits):
        self.set_n_qubits(n_qubits)

    def set_n_qubits(self, n_qubits):
        self.n_qubits = n_qubits
        self.dim = 2 ** n_qubits
        self.u_basis = np.array(get_diracs(n_qubits))
        a = ProcessFidelityFinder.one_qubit_a
        n_qubit_pure_basis = ProcessFidelityFinder.pure_state_basis
        for _ in range(n_qubits - 1):
            n_qubit_pure_basis = [np.kron(x, y) for x in ProcessFidelityFinder.pure_state_basis for y in n_qubit_pure_basis]
            a = np.kron(a, ProcessFidelityFinder.one_qubit_a)
        self.a = a
        self.n_qubit_pure_basis = n_qubit_pure_basis
        self.pauli_strings = list(product("IXYZ", repeat=n_qubits))
        self.preps = [self.prepare_state(t) for t in product([0, 1, 2, 3], repeat=n_qubits)]

    def f_pro(self, channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
        dim = channel[0].shape[0]
        n_qubits = int(math.log(dim, 2))  # please don't give me qudits, the Pauli's aren't nice
        u_basis = get_diracs(n_qubits)
        state_basis = ProcessFidelityFinder.simple_state_basis
        for _ in range(n_qubits - 1):
            state_basis = [np.kron(x, y) for x in state_basis for y in ProcessFidelityFinder.simple_state_basis]
        a = np.array(np.eye(dim ** 2) -
                     np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
        sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                       for k in range(dim ** 2)]) for l in range(dim ** 2)]
        return 1 / dim ** 3 * \
            sum([np.trace(sigmas[k] @ apply_channel(channel, state_basis[k])) for k in range(dim ** 2)]).real

    def f_pro_experimental(self, circuit_string: Iterable[Any], unitary: np.ndarray, noise_channels: Iterable = [],
                           key: Dict[Any, np.ndarray] = None) -> float:
        dim = unitary.shape[0]
        n_qubits = int(math.log(dim, 2))
        u_basis = get_diracs(n_qubits)
        state_basis = ProcessFidelityFinder.simple_state_basis
        for _ in range(n_qubits - 1):
            state_basis = [np.kron(x, y) for x in state_basis for y in ProcessFidelityFinder.simple_state_basis]
        a = np.array(np.eye(dim ** 2) -
                     np.outer([0] + [1 for _ in range(dim ** 2 - 1)], [1] + [0 for _ in range(dim ** 2 - 1)]))
        sigmas = [sum([a[k][l] * unitary @ u_basis[k] @ unitary.transpose().conjugate()
                       for k in range(dim ** 2)]) for l in range(dim ** 2)]
        expectations = [np.trace(sigmas[k] @ density_runner.run_by_matrices(circuit_string, state_basis[k], noise_channels, key))
                        .real for k in range(dim ** 2)]
        # print(expectations)
        return 1 / dim ** 3 * sum(expectations)

    def prepare_state(self, single_qubit_states):
        circuit = Circuit(self.n_qubits)
        for i in range(len(single_qubit_states)):
            ProcessFidelityFinder.one_qubit_preps[single_qubit_states[i]](i, circuit)
        return circuit

    def load_expectations_from_file(self, filename):
        expectations = np.full((self.dim ** 2,), np.nan)
        with open(filename) as file:
            lines = file.readlines()
        for line in lines:
            d = line.split()
            expectations[int(d[0])] = float(d[1])
        return expectations

    def save_expectations_to_file(self, filename, expectations: np.ndarray):
        print("saving to file", expectations)
        with open(filename, "a+") as file:
            file.writelines([str(k) + " " + str(expectations[k]) + "\n"
                             for k in range(self.dim ** 2) if not np.isnan(expectations[k])])

    def f_pro_simulated(self, circuit: Circuit, unitary: np.ndarray, backend: Backend, filename, start=0) -> float:
        b = np.einsum('ab,jbc,dc,lda->lj', unitary, self.u_basis, unitary.conjugate(), self.u_basis, optimize=True) / self.dim
        m = (b @ self.a).real
        expectations = self.load_expectations_from_file(filename)
        for k in range(start, self.dim ** 2):
            if np.isnan(expectations[k]):
                sigma = QubitOperator()
                for l in range(self.dim ** 2):
                    pauli_string = "".join(self.pauli_strings[l])
                    sigma += QubitOperator([(i, pauli_string[i]) for i in range(len(pauli_string)) if pauli_string[i] != "I"], m[l][k])
                c = self.preps[k].copy()
                c.add_circuit(circuit, list(range(self.n_qubits)))
                expectations[k] = backend.get_operator_expectation_value(c, sigma, shots=8192)
                self.save_expectations_to_file(filename, expectations)
        return 1 / self.dim ** 3 * sum(expectations).real

    def effect_of_noise(self, circuit_description: Iterable[Any], noise_channels: Iterable = [],  n_qubits: int = 1,
                        circuit_key: Optional[Dict[Any, np.ndarray]] = None) -> float:
        d = 2 ** n_qubits
        unitary = np.eye(d)
        for s in circuit_description[::-1]:
            unitary = circuit_key[s] @ unitary
        return 1 - self.f_pro_experimental(circuit_description, unitary, noise_channels, key=circuit_key)

    def angle(self, channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
        return math.acos(self.f_pro(channel, unitary) ** 0.5)

    def bures(self, channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:
        f = max(self.f_pro(channel, unitary), 0)
        return (1 - f ** 0.5) ** 0.5

    def C(self, channel: Iterable[np.ndarray], unitary: np.ndarray) -> float:  # That's the only name they give it
        return (1 - self.f_pro(channel, unitary)) ** 0.5
