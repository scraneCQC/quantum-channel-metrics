from functools import reduce
from typing import Iterable, Dict
from Pauli import *
from caching_runner import CachingRunner
from J_fidelity import f_pro


class SubcircuitRemover:

    def __init__(self, circuit_description: Iterable, circuit_key: Dict, noise_channels: Iterable, n_qubits: int = 1,
                 verbose: bool = False):
        if verbose:
            print("Setting up subcircuit remover...")
        self.circuit = circuit_description
        self.key = circuit_key
        self.gates = list(self.key.keys())
        self.noise_channels = noise_channels
        self.n_qubits = n_qubits
        self.verbose = verbose
        self.u_basis = get_diracs(self.n_qubits)
        self.state_basis = np.array(self.u_basis)
        self.d2 = 2 ** (2 * self.n_qubits)
        self.U = reduce(lambda x, y: x @ y, [self.key[s] for s in self.circuit], np.eye(2 ** self.n_qubits))
        self.sigmas = np.array([self.U @ u @ self.U.transpose().conjugate() for u in self.u_basis])
        self.runner = CachingRunner(circuit_key, n_qubits, noise_channels)
        self.runner.remember(self.circuit)
        [self.runner.remember([x, y]) for x in circuit_key.keys() for y in circuit_key.keys()]
        if verbose:
            print("Ready to optimize your noisy circuits")

    def set_circuit(self, circuit):
        self.circuit = circuit
        self.set_target_unitary(reduce(lambda x, y: x @ y, [self.key[s] for s in self.circuit], np.eye(2 ** self.n_qubits)))

    def set_target_unitary(self, unitary):
        self.U = unitary
        self.sigmas = np.array([self.U @ u @ self.U.transpose().conjugate() for u in self.u_basis])

    def should_remove_subcircuit(self, start: int, end: int, original_fid: float) -> bool:
        remove_fid = self.fidelity(self.circuit[:start] + self.circuit[end:])
        if remove_fid >= original_fid:
            if self.verbose:
                print("removing", self.circuit[start:end], "for a gain of", remove_fid - original_fid)
            self.circuit = self.circuit[:start] + self.circuit[end:]
            return True
        return False

    def should_replace_subcircuit(self, start: int, end: int, original_fid: float) -> bool:
        replace_fid = [([g], self.fidelity(self.circuit[:start] + [g] + self.circuit[end:])) for g in self.gates]
        g, f = max(replace_fid, key=lambda x: x[1])
        if f - original_fid > 1e-5:
            if self.verbose:
                print("replacing", self.circuit[start:end], "with", g, "for a gain of", f - original_fid)
            self.circuit = self.circuit[:start] + g + self.circuit[end:]
            return True
        return False

    def should_replace_with_2(self, start: int, end: int, original_fid: float) -> bool:
        replace_fid = [([g, h], self.fidelity(self.circuit[:start] + [g, h] + self.circuit[end:])) for g in self.gates for h in self.gates]
        g, f = max(replace_fid, key=lambda x: x[1])
        if f - original_fid > 1e-5:
            if self.verbose:
                print("replacing", self.circuit[start:end], "with", g, "for a gain of", f - original_fid)
            self.circuit = self.circuit[:start] + g + self.circuit[end:]
            return True
        return False

    def remove_any_subcircuit(self) -> bool:
        length = len(self.circuit)
        original_fid = self.fidelity(self.circuit)
        return any(self.should_remove_subcircuit(s, s + i, original_fid)
                   for i in range(1, min(6, length + 1)) for s in range(length - i + 1))

    def replace_any_subcircuit(self) -> bool:
        length = len(self.circuit)
        if length == 1:
            return False
        original_fid = self.fidelity(self.circuit)
        return any(self.should_replace_subcircuit(s, s + i, original_fid)
                   for i in range(1, min(6, length + 1)) for s in range(length - i + 1))

    def replace_any_with2(self) -> bool:
        length = len(self.circuit)
        if length == 1:
            return False
        original_fid = self.fidelity(self.circuit)
        return any(self.should_replace_with_2(s, s + i, original_fid)
                   for i in range(3, min(6, length + 1)) for s in range(length - i + 1))

    def reorder(self) -> bool:
        l = len(self.circuit)
        for s in range(l - 1):
            name1 = self.circuit[s]
            if name1[0] != "c" and self.circuit[s + 1][0] != "c":
                # single qubit gates
                i = int(self.circuit[s][2])
                j = int(self.circuit[s + 1][2])
                if i > j:
                    self.circuit[s] = self.circuit[s + 1]
                    self.circuit[s + 1] = name1
                    if self.verbose:
                        print("Re-ordering single qubit gates")
                    return True
        return False

    def reduce_circuit(self) -> Iterable:
        while self.remove_any_subcircuit() or self.reorder() or self.replace_any_subcircuit(): # or self.replace_any_with2():
            pass
        return self.circuit

    def fidelity(self, circuit_string: Iterable) -> float:
        s = np.einsum('kij,lk,lji->', self.sigmas, self.runner.get_matrix(circuit_string), self.state_basis, optimize=True).real
        return 2 ** (-3 * self.n_qubits) * s

    @property
    def unitary(self) -> np.ndarray:
        return reduce(lambda x, y: x @ y, (self.key[s] for s in self.circuit), np.eye(2 ** self.n_qubits))


def run_all(circuits: Iterable, circuit_key: Dict, noise_channels: Iterable, n_qubits: int = 1, verbose: bool = False):
    if len(circuits) == 0:
        return []
    sr = SubcircuitRemover(circuits[0], circuit_key, noise_channels, n_qubits, verbose)
    res = []
    for c in circuits:
        sr.set_circuit(c)
        if verbose:
            print("Running on circuit:\n", c)
            print("Original fidelity:", sr.fidelity(c))
            u1 = sr.U
        res.append(sr.reduce_circuit())
        if verbose:
            u2 = sr.unitary
            print("Result:\n", res[-1])
            print("New fidelity:", sr.fidelity(sr.circuit))
            print("Fidelity of ideal circuits", f_pro([u1], u2))
            print()
            print()
    return res
