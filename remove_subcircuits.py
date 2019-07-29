from functools import reduce
from density_runner import run_by_matrices
from typing import Iterable, Dict, Any
from Pauli import *


class SubcircuitRemover:

    def __init__(self, circuit_description, circuit_key, noise_channels, n_qubits=1, verbose=False):
        self.circuit = circuit_description
        self.key = circuit_key
        self.gates = list(self.key.keys())
        self.noise_channels = noise_channels
        self.n_qubits = n_qubits
        self.verbose = verbose
        self.u_basis = get_diracs(self.n_qubits)
        one_qubit_state_basis = [I1, I1 + X, I1 + Y, I1 + Z]
        self.state_basis = one_qubit_state_basis
        for _ in range(self.n_qubits - 1):
            self.state_basis = [np.kron(x, y) for x in self.state_basis for y in one_qubit_state_basis]
        self.d2 = 2 ** (2 * self.n_qubits)
        self.a = np.array(np.eye(self.d2) -
                          np.outer([0] + [1 for _ in range(self.d2 - 1)], [1] + [0 for _ in range(self.d2 - 1)]))
        self.U = reduce(lambda x, y: x @ y, [self.key[s] for s in self.circuit], np.eye(2 ** self.n_qubits))
        self.sigmas = [sum([self.a[k][l] * self.U @ self.u_basis[k] @ self.U.transpose().conjugate()
                       for k in range(self.d2)]) for l in range(self.d2)]
        #self.runner = CachingRunner(circuit_key, n_qubits, noise_channels)
        #self.runner.cache.update({"".join(self.circuit): self.runner.circuit_process_matrix(self.circuit)})

    def _get_unitary(self, desc):
        return reduce(lambda x, y: x @ y, [self.key[s] for s in desc], np.eye(2 ** self.n_qubits))

    def should_remove_subcircuit(self, start, end, original_fid):
        remove_fid = self.f_pro_experimental(self.circuit[:start] + self.circuit[end:])
        if remove_fid >= original_fid:
            if self.verbose:
                print("removing", self.circuit[start:end])
            self.circuit = self.circuit[:start] + self.circuit[end:]
            return True
        return False

    def should_replace_subcircuit(self, start, end, original_fid):
        replace_fid = [(g, self.f_pro_experimental(self.circuit[:start] + [g] + self.circuit[end:])) for g in self.gates]
        g, f = max(replace_fid, key=lambda x: x[1])
        if f > original_fid:
            if self.verbose:
                print("replacing", self.circuit[start:end], "with", [g])
            self.circuit = self.circuit[:start] + [g] + self.circuit[end:]
            return True
        return False

    def remove_any_subcircuit(self):
        length = len(self.circuit)
        original_fid = self.f_pro_experimental(self.circuit)
        return any(self.should_remove_subcircuit(s,s+i, original_fid)
                   for i in range(1, min(6, length + 1)) for s in range(length - i + 1))

    def replace_any_subcircuit(self):
        length = len(self.circuit)
        if length == 1:
            return False
        original_fid = self.f_pro_experimental(self.circuit)
        return any(self.should_replace_subcircuit(s, s+i, original_fid)
                   for i in range(min(5, length), 0, -1) for s in range(length - i + 1))

    def reduce_circuit(self):
        while self.remove_any_subcircuit() or self.replace_any_subcircuit():
            pass
        return self.circuit

    def f_pro_experimental(self, circuit_string: Iterable[Any]) -> float:
        expectations = (
            np.trace(self.sigmas[k] @ run_by_matrices(circuit_string, self.state_basis[k], self.noise_channels, self.key))
                .real for k in range(self.d2))
        return 2 ** (-3 * self.n_qubits) * sum(expectations)

    def fid_with_id(self, unitary):
        sigmas = (sum(self.a[k][l] * unitary @ self.u_basis[k] @ unitary.transpose().conjugate()
                      for k in range(self.d2)) for l in range(self.d2))
        return 2 ** (-3 * self.n_qubits) * sum(np.trace(sigmas[k] @ self.state_basis[k]) for k in range(self.d2)).real

    @property
    def unitary(self):
        return reduce(lambda x, y: x @ y, (self.key[s] for s in self.circuit), np.eye(2 ** self.n_qubits))
