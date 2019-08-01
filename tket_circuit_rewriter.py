from pytket import Circuit, Transform, OpType
from common_gates import multi_qubit_matrix, H, Rx, Ry, Rz, U3, cnot
from Pauli import X, Y, Z, get_diracs
from functools import reduce
import numpy as np

# noinspection PyCallByClass
cleanup = Transform.sequence([Transform.RebaseToRzRx(),
                              Transform.repeat(Transform.sequence([
                                  Transform.RemoveRedundancies(),
                                  Transform.ReduceSingles(),
                                  Transform.CommuteRzRxThroughCX()]))])


matrices_no_params = {OpType.Z: lambda i, n: multi_qubit_matrix(Z, i[0], n),
                      OpType.X: lambda i, n: multi_qubit_matrix(X, i[0], n),
                      OpType.Y: lambda i, n: multi_qubit_matrix(Y, i[0], n),
                      OpType.H: lambda i, n: multi_qubit_matrix(H, i[0], n),
                      OpType.CX: lambda i, n: cnot(i[0], i[1], n)}

matrices_with_params = {OpType.Rx: lambda i, n, params: Rx(params[0], i[0], n),
                        OpType.Ry: lambda i, n, params: Ry(params[0], i[0], n),
                        OpType.Rz: lambda i, n, params: Rz(params[0], i[0], n),
                        OpType.U1: lambda i, n, params: Rz(params[0], i[0], n),
                        OpType.U3: lambda i, n, params: U3(params[0], params[1], params[2], i[0], n)}


class RewriteTket:

    def __init__(self, circuit: Circuit, noise_channels, target=None, verbose=False):
        self.verbose = verbose
        if cleanup.apply(circuit) and self.verbose:
            print("Cleaned circuit up:")
            print(circuit.get_commands())
        self.set_circuit(circuit)
        self.noise_channels = noise_channels
        self.target = target
        if target is None:
            self.target = self.matrix_list_product(self.instructions_to_matrices(self.instructions))
        self.u_basis = get_diracs(self.n_qubits)
        self.state_basis = np.array(self.u_basis)
        self.d2 = 2 ** (2 * self.n_qubits)
        self.sigmas = np.array([self.target @ u @ self.target.transpose().conjugate() for u in self.u_basis])
        self.noise_process = self.matrix_list_product([self.get_individual_process_matrix(c) for c in noise_channels])
        if self.verbose:
            print("original fidelity is", self.fidelity(self.instructions))

    def set_circuit(self, circuit: Circuit):
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.instructions = circuit.get_commands()

    def set_target_unitary(self, target: np.ndarray):
        self.target = target
        self.sigmas = np.array([self.target @ u @ self.target.transpose().conjugate() for u in self.u_basis])

    def matrix_list_product(self, matrices, default_size=2):
        if len(matrices) == 0:
            return np.eye(default_size)
        return reduce(lambda x, y: x @ y, matrices, np.eye(matrices[0].shape[0]))

    def should_remove(self, index, original_fidelity):
        new_circuit = self.instructions[:index] + self.instructions[index+1:]
        new_circuit = self.instructions_to_circuit(new_circuit)
        cleanup.apply(new_circuit)
        new_fidelity = self.fidelity(new_circuit.get_commands())
        if new_fidelity > original_fidelity:
            return new_fidelity - original_fidelity, new_circuit, index
        return (-1, -1, index)

    def remove_any(self):
        original_fidelity = self.fidelity(self.instructions)
        diffs = [self.should_remove(i, original_fidelity) for i in range(len(self.instructions)) if self.instructions[i].op.get_type() != OpType.CX]
        if len(diffs) == 0:
            return False
        f, c, i = max(diffs, key=lambda x: x[0])
        if f > 1e-5:
            print("Removing", self.instructions[i], "to improve fidelity by", f)
            self.set_circuit(c)
            return True
        return False

    def should_commute(self, index, original_fidelity):
        gate1 = self.instructions[index]
        gate2 = self.instructions[index + 1]
        new_circuit = self.instructions[:index] + [gate2, gate1] + self.instructions[index + 2:]
        new_circuit = self.instructions_to_circuit(new_circuit)
        cleanup.apply(new_circuit)
        new_fidelity = self.fidelity(new_circuit.get_commands())
        if new_fidelity > original_fidelity:
            return new_fidelity - original_fidelity, new_circuit, index
        return (-1, None, None)

    def commute_any(self):
        if len(self.instructions) < 2:
            return False
        original_fidelity = self.fidelity(self.instructions)
        diffs = [self.should_commute(i, original_fidelity) for i in range(len(self.instructions) - 1)]
        f, c, i = max(diffs, key=lambda x: x[0])
        if f > 1e-5:
            print("Commuting", self.instructions[i], "with", self.instructions[i + 1], "to improve fidelity by", f)
            self.set_circuit(c)
            return True
        return False

    def should_change_angle(self, index):
        return False

    def instructions_to_circuit(self, instructions):
        c = Circuit(self.n_qubits)
        for inst in instructions:
            t = inst.op.get_type()
            c.add_operation(t, inst.op.get_params(), inst.qubits)
        return c

    def instructions_to_matrices(self, instructions):
        matrices = []
        for inst in instructions:
            t = inst.op.get_type()
            if t in matrices_no_params:
                matrices.append(matrices_no_params[t](inst.qubits, self.n_qubits))
            elif t in matrices_with_params:
                matrices.append(matrices_with_params[t](inst.qubits, self.n_qubits, inst.op.get_params()))
            else:
                raise ValueError("Unexpected instruction", inst)
        return matrices

    def get_individual_process_matrix(self, channel):
        return np.vstack([[np.trace(sum(e @ d2 @ e.transpose().conjugate() for e in channel) @ d1) / (2 ** self.n_qubits) for d1 in self.u_basis]
                          for d2 in self.u_basis]).transpose()

    def get_process_matrix(self, instructions):
        return self.matrix_list_product([self.noise_process @ self.get_individual_process_matrix([m]) for m in self.instructions_to_matrices(instructions)], default_size=self.d2)

    def fidelity(self, instructions):
        s = np.einsum('kij,lk,lji->', self.sigmas, self.get_process_matrix(instructions), self.state_basis, optimize=True).real
        return 2 ** (-3 * self.n_qubits) * s

    def reduce(self):
        applied = False
        while self.remove_any() or self.commute_any():
            applied = True
        if self.verbose:
            if applied:
                print("New fidelity is", self.fidelity(self.instructions))
            else:
                print("Didn't find anything to improve")
        return applied
