from Pauli import get_diracs, one_qubit_diracs
from common_gates import cnot12, cnot21, swap
from pytket import OpType
import numpy as np
import math
import scipy.sparse as sp
from tket_pauli_gadgets.converter import matrices_no_params, matrices_with_params, converter


class ProcessMatrixFinder:
    def __init__(self, n_qubits, one_qubit_noise_channels, cnot_noise_channels):
        self.process_cache = dict()
        self.two_qubit_diracs = get_diracs(2)
        self.n_qubits = n_qubits
        self.use_sparse = n_qubits > 4
        self.d2 = 4 ** n_qubits
        self.basic_cnot_processes = {1: self.get_adjacent_cnot_process_matrix(0, 1),
                                     -1: self.get_adjacent_cnot_process_matrix(1, 0)}
        self.cnot_noise_process = converter.matrix_list_product(
            [self.get_two_qubit_noise_process(c) for c in cnot_noise_channels], default_size=16)
        self.single_qubit_gate_noise_process = converter.matrix_list_product(
            [self.get_single_qubit_noise_process(c) for c in one_qubit_noise_channels], default_size=4)
        self.cnot_processes = {(i, j): self.get_cnot_process_matrix(i, j)
                               for i in range(self.n_qubits) for j in range(self.n_qubits) if i != j}
        self.swap_processes = dict()

    def get_adjacent_cnot_process_matrix(self, control, target):
        if target == control + 1:
            c = cnot12
        elif control == target + 1:
            c = cnot21
        else:
            raise ValueError("Please use get_cnot_process_matrix instead", control, target)
        little_process = np.vstack([[np.einsum('ij,ji->', c @ d2 @ c, d1, optimize=True) / 4
                                     for d1 in self.two_qubit_diracs] for d2 in self.two_qubit_diracs]).transpose()
        return little_process

    def get_cnot_process_matrix(self, control, target):
        if control == target:
            raise ValueError("Control and target must be different")
        elif target > control:
            g = self.basic_cnot_processes[1]
        else:
            g = self.basic_cnot_processes[-1]
        g = self.cnot_noise_process @ g
        d = abs(control - target)
        m = min(control, target)
        if self.use_sparse:
            if d > 1:
                g =np.kron(np.eye(4 ** (d - 1)), g)
                g = np.moveaxis(g.reshape((4, 4) * (d + 1)), [0, d - 1, d + 1, 2 * d], [d - 1, 0, 2 * d, d + 1])\
                    .reshape((4 ** (d + 1), 4 ** (d + 1)))
            g = sp.csr_matrix(g)
            z = sp.kron(sp.eye(4 ** m), sp.kron(g, sp.eye(4 ** (self.n_qubits - 1 - max(control, target)))))
        else:
            if d > 1:
                g = np.kron(np.eye(4 ** (d - 1)), g)
                g = np.moveaxis(g.reshape((4, 4) * (d + 1)), [0, d - 1, d + 1, 2 * d], [d - 1, 0, 2 * d, d + 1])\
                    .reshape((4 ** (d + 1), 4 ** (d + 1)))
            z = np.kron(np.eye(4 ** m), np.kron(g, np.eye(4 ** (self.n_qubits - 1 - max(control, target)))))
        return z

    def get_single_qubit_noise_process(self, noise_channel):
        little_process = np.vstack([[np.einsum('ij,ji->',
                            sum([e @ d2 @ e.transpose().conjugate() for e in noise_channel]), d1, optimize=True) / 2
                                     for d1 in one_qubit_diracs] for d2 in one_qubit_diracs]).transpose()
        return little_process

    def get_two_qubit_noise_process(self, noise_channel):
        little_process = np.vstack([[np.einsum('ij,ji->',
                            sum([e @ d2 @ e.transpose().conjugate() for e in noise_channel]), d1, optimize=True) / 4
                                     for d1 in self.two_qubit_diracs] for d2 in self.two_qubit_diracs]).transpose()
        return little_process

    def get_single_qubit_gate_process_matrix(self, instruction):
        s = str(instruction)
        #if s in self.process_cache:
        #    return self.process_cache[s]
        t = instruction.op.get_type()
        qubit = instruction.qubits[0]
        if t in matrices_no_params:
            gate = matrices_no_params[t]([0], 1)
        elif t in matrices_with_params:
            gate = matrices_with_params[t]([0], 1, [p * math.pi for p in instruction.op.get_params()])
        else:
            raise ValueError("Unexpected instruction", instruction)
        little_process = self.single_qubit_gate_noise_process @ \
            np.vstack([[np.einsum('ij,ji->', gate @ d2 @ gate.transpose().conjugate(), d1, optimize=True) / 2
                        for d1 in one_qubit_diracs] for d2 in one_qubit_diracs]).transpose()
        if self.use_sparse:
            z = sp.kron(sp.kron(sp.eye(4 ** qubit), sp.csr_matrix(little_process)), sp.eye(4 ** (self.n_qubits - qubit - 1)))
        else:
            z = np.kron(np.kron(np.eye(4 ** qubit), little_process), np.eye(4 ** (self.n_qubits - qubit - 1)))
        # self.process_cache.update({s: z})
        return z

    def get_swap_process_matrix(self, i, j):
        if (i, j) in self.swap_processes:
            return self.swap_processes[(i, j)]
        elif (j, i) in self.swap_processes:
            return self.swap_processes[(j, i)]
        else:
            p = self.unitary_to_process_matrix(swap(i, j, self.n_qubits))
            self.swap_processes[(i, j)] = p
            return p

    def unitary_to_process_matrix(self, unitary, n=None):
        if n is None:
            n = self.n_qubits
        return np.vstack([[np.einsum('ij,ji->', unitary @ d2 @ unitary.transpose().conjugate(), d1, optimize=True) / (2 ** n)
                        for d1 in get_diracs(n)] for d2 in get_diracs(n)]).transpose()

    def instructions_to_process_matrix(self, instructions):
        s = "".join([str(inst) for inst in instructions])
        #if s in self.process_cache:
        #    return self.process_cache[s]
        #l = len(instructions)
        #for i in range(len(instructions)):
        #    end = "".join([str(inst) for inst in instructions[i:]])
        #    if end in self.process_cache:
        #        m = self.process_cache[end] @ self.instructions_to_process_matrix(instructions[:i])
        #        if l < 5:
        #            self.process_cache.update({s: m})
        #        return m
        #    beginning = "".join([str(inst) for inst in instructions[:(l - i)]])
        #    if beginning in self.process_cache:
        #        m = self.instructions_to_process_matrix(instructions[(l - i):]) @ self.process_cache[beginning]
        #        if l < 5:
        #            self.process_cache.update({s: m})
        #        return m
        m = np.eye(self.d2)
        #m = converter.matrix_list_product([self.cnot_processes[tuple(inst.qubits)] if inst.op.get_type() == OpType.CX else
        #                                   self.get_single_qubit_gate_process_matrix(inst) for inst in instructions], default_size=self.d2)
        for inst in instructions:
            if inst.op.get_type() == OpType.CX:
                m = self.cnot_processes[tuple(inst.qubits)] @ m
            else:
                m = self.get_single_qubit_gate_process_matrix(inst) @ m
        #if l < 5:
        #    self.process_cache.update({s: m})
        return m

