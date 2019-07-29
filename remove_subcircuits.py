from functools import reduce
import numpy as np
from density_runner import run_by_matrices, apply_channel
from typing import Iterable, Dict, Any
import math
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

    def _get_unitary(self, desc):
        return reduce(lambda x, y: x @ y, [self.key[s] for s in desc], np.eye(2 ** self.n_qubits))

    def should_remove_subcircuit(self, start, end):
        include_subcircuit = self.f_pro_experimental(self.circuit, self.U)
        remove_subcircuit = self.f_pro_experimental(self.circuit[:start] + self.circuit[end:], self.U)
        return remove_subcircuit >= include_subcircuit

    def should_replace_subcircuit(self, start, end):
        replace_fid = [(g, self.f_pro_experimental(self.circuit[:start] + [g] + self.circuit[end:], self.U)) for g in self.gates]
        original_fid = self.f_pro_experimental(self.circuit, self.U)
        g, f = max(replace_fid, key=lambda x: x[1])
        if f > original_fid:
            return [g]
        return False

    def remove_any_subcircuit(self):
        length = len(self.circuit)
        for i in range(1, min(6, length + 1)):  # Test subcircuits of length i
            for s in range(length - i + 1):  # That start at s
                e = s + i
                res = self.should_remove_subcircuit(s, s + i)
                if res:
                    if self.verbose:
                        print("removing", self.circuit[s:e])
                    self.circuit = self.circuit[:s] + self.circuit[e:]
                    return True
        return False

    def replace_any_subcircuit(self):
        length = len(self.circuit)
        if length == 1:
            return False
        for i in range(min(5, length), 1, -1):  # Test subcircuits of length i
            for s in range(length - i + 1):  # That start at s
                e = s + i
                res = self.should_replace_subcircuit(s, e)
                if res:
                    if self.verbose:
                        print("replacing", self.circuit[s:e], "with", res)
                    self.circuit = self.circuit[:s] + res + self.circuit[e:]
                    return True
        return False

    def reduce_circuit(self):
        while self.remove_any_subcircuit() or self.replace_any_subcircuit():
            pass
        return self.circuit

    def f_pro_experimental(self, circuit_string: Iterable[Any], unitary: np.ndarray) -> float:
        sigmas = [sum([self.a[k][l] * unitary @ self.u_basis[k] @ unitary.transpose().conjugate()
                       for k in range(self.d2)]) for l in range(self.d2)]
        expectations = [
            np.trace(sigmas[k] @ run_by_matrices(circuit_string, self.state_basis[k], self.noise_channels, self.key))
            .real for k in range(self.d2)]
        return 2 ** (-3 * self.n_qubits) * sum(expectations)

    def fid_with_id(self, unitary):
        sigmas = [sum([self.a[k][l] * unitary @ self.u_basis[k] @ unitary.transpose().conjugate()
                       for k in range(self.d2)]) for l in range(self.d2)]
        return 2 ** (-3 * self.n_qubits) * sum([np.trace(sigmas[k] @ self.state_basis[k]) for k in range(self.d2)]).real

    @property
    def unitary(self):
        return reduce(lambda x, y: x @ y, [self.key[s] for s in self.circuit], np.eye(2 ** self.n_qubits))