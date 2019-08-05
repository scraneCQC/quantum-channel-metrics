from pytket import Circuit, Transform, OpType
from common_gates import multi_qubit_matrix, H, Rx, Ry, Rz, U1, U3, cnot, cnot12, cnot21, S, V
from Pauli import X, Y, Z, get_diracs, one_qubit_diracs
from functools import reduce
import numpy as np
import math

# noinspection PyCallByClass
cleanup = Transform.sequence([
                              Transform.repeat(Transform.sequence([
                                  Transform.RemoveRedundancies(),
                                  Transform.ReduceSingles(),
                                  Transform.CommuteRzRxThroughCX()])),
                              Transform.OptimisePauliGadgets(),
                              Transform.ReduceSingles()])


matrices_no_params = {OpType.Z: lambda i, n: multi_qubit_matrix(Z, i[0], n),
                      OpType.X: lambda i, n: multi_qubit_matrix(X, i[0], n),
                      OpType.Y: lambda i, n: multi_qubit_matrix(Y, i[0], n),
                      OpType.H: lambda i, n: multi_qubit_matrix(H, i[0], n),
                      OpType.S: lambda i, n: multi_qubit_matrix(S, i[0], n),
                      OpType.V: lambda i, n: multi_qubit_matrix(V, i[0], n),
                      OpType.CX: lambda i, n: cnot(i[0], i[1], n)}

matrices_with_params = {OpType.Rx: lambda i, n, params: Rx(params[0], i[0], n),
                        OpType.Ry: lambda i, n, params: Ry(params[0], i[0], n),
                        OpType.Rz: lambda i, n, params: Rz(params[0], i[0], n),
                        OpType.U1: lambda i, n, params: U1(params[0], i[0], n),
                        OpType.U3: lambda i, n, params: U3(params[0], params[1], params[2], i[0], n)}


class RewriteTket:

    def __init__(self, circuit: Circuit, noise_channels, cnot_noise_channels, target=None, verbose=False):
        self.verbose = verbose
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.instructions = circuit.get_commands()
        self.target = target
        if target is None:
            self.target = self.matrix_list_product([self.instruction_to_unitary(inst) for inst in self.instructions[::-1]])
        self.two_qubit_diracs = get_diracs(2)
        self.u_basis = get_diracs(self.n_qubits)
        self.state_basis = np.array(self.u_basis)
        self.d2 = 2 ** (2 * self.n_qubits)
        self.sigmas = np.array([self.target @ u @ self.target.transpose().conjugate() for u in self.u_basis])
        self.contracted = np.einsum('kij,lji->kl', self.sigmas, self.state_basis, optimize=True)
        self.noise_process = self.matrix_list_product(
            [self.get_single_qubit_noise_process(c) for c in noise_channels], default_size=4)
        self.cnot_noise = self.matrix_list_product(
            [self.get_two_qubit_noise_process(c) for c in cnot_noise_channels], default_size=16)
        self.basic_cnot_processes = {1: self.get_adjacent_cnot_process_matrix(0,1),
                                     -1: self.get_adjacent_cnot_process_matrix(1,0)}
        self.cnot_processes = {(i, j): self.get_cnot_process_matrix(i, j)
                               for i in range(self.n_qubits) for j in range(self.n_qubits) if i != j}
        self.process_cache = dict()
        self.original_fidelity = -1

    def set_circuit(self, circuit: Circuit):
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.instructions = circuit.get_commands()

    def set_target_unitary(self, target: np.ndarray):
        self.target = target
        self.sigmas = np.array([self.target @ u @ self.target.transpose().conjugate() for u in self.u_basis])
        self.contracted = np.einsum('kij,lji->kl', self.sigmas, self.state_basis, optimize=True)

    def set_circuit_and_target(self, circuit):
        self.set_circuit(circuit)
        self.set_target_unitary(self.matrix_list_product(
            [self.instruction_to_unitary(inst) for inst in self.instructions[::-1]]))

    def matrix_list_product(self, matrices, default_size=None):
        if len(matrices) == 0:
            if default_size is None:
                default_size = 2 ** self.n_qubits
            return np.eye(default_size)
        return reduce(lambda x, y: x @ y, matrices, np.eye(matrices[0].shape[0]))

    def should_remove(self, index):
        new_circuit = self.instructions[:index] + self.instructions[index+1:]
        new_circuit = self.instructions_to_circuit(new_circuit)
        cleanup.apply(new_circuit)
        new_fidelity = self.fidelity(new_circuit.get_commands())
        if new_fidelity > self.original_fidelity:
            return new_fidelity - self.original_fidelity, new_circuit, index
        return -1, -1, index

    def remove_any(self):
        if self.n_qubits < 6:
            diffs = [self.should_remove(i) for i in range(len(self.instructions))
                     if self.instructions[i].op.get_type() != OpType.CX]
        else:
            diffs = []
            for i in range(len(self.instructions)):
                if self.instructions[i].op.get_type() != OpType.CX:
                    diffs.append(self.should_remove(i))
        if len(diffs) == 0:
            return False
        f, c, i = max(diffs, key=lambda x: x[0])
        if f > 1e-5:
            if self.verbose:
                print("Removing", self.instructions[i], "to improve fidelity by", f)
            self.set_circuit(c)
            self.original_fidelity = self.original_fidelity + f
            return True
        return False

    def should_commute(self, index):
        gate1 = self.instructions[index]
        gate2 = self.instructions[index + 1]
        new_circuit = self.instructions[:index] + [gate2, gate1] + self.instructions[index + 2:]
        new_circuit = self.instructions_to_circuit(new_circuit)
        cleanup.apply(new_circuit)
        new_fidelity = self.fidelity(new_circuit.get_commands())
        if new_fidelity > self.original_fidelity:
            return new_fidelity - self.original_fidelity, new_circuit, index
        return -1, None, None

    def commute_any(self):
        if len(self.instructions) < 2:
            return False
        if self.n_qubits < 6:
            diffs = [self.should_commute(i) for i in range(len(self.instructions) - 1)]
        else:
            diffs = []
            for i in range(len(self.instructions) - 1):
                diffs. append((self.should_commute(i)))
        f, c, i = max(diffs, key=lambda x: x[0])
        if f > 1e-5:
            if self.verbose:
                print("Commuting", self.instructions[i], "with", self.instructions[i + 1], "to improve fidelity by", f)
            self.set_circuit(c)
            self.original_fidelity = self.original_fidelity + f
            return True
        return False

    def should_change_angle(self, index):
        inst = self.instructions[index]
        if inst.op.get_type() in matrices_with_params:
        #if inst.op.get_type() == OpType.U1:
            params = inst.op.get_params()
            for i in range(len(params)):
                if params[i] not in [0, 0.5, 1, 1.5]:
                    new_params = list(params)
                    new_params[i] = round(params[i] * 2) / 2
                    new_circuit = self.instructions_to_circuit(self.instructions[:index])
                    new_circuit.add_operation(inst.op.get_type(), new_params, inst.qubits)
                    new_circuit.add_circuit(self.instructions_to_circuit(self.instructions[index + 1:]), list(range(self.n_qubits)))
                    cleanup.apply(new_circuit)
                    new_fidelity = self.fidelity(new_circuit.get_commands())
                    if new_fidelity > self.original_fidelity:
                        return new_fidelity - self.original_fidelity, new_circuit, index, new_params
        return -1, None, None, None

    def change_any_angle(self):
        if self.n_qubits < 6:
            diffs = [self.should_change_angle(i) for i in range(len(self.instructions))]
        else:
            diffs = []
            for i in range(len(self.instructions) - 1):
                diffs. append((self.should_commute(i)))
        if len(diffs) == 0:
            return False
        f, c, i, p = max(diffs, key=lambda x: x[0])
        if f > 1e-5:
            if self.verbose:
                print("Changing angle of", self.instructions[i], "to", p , "to improve fidelity by", f)
            self.set_circuit(c)
            self.original_fidelity = self.original_fidelity + f
            return True
        return False


    def instructions_to_circuit(self, instructions):
        c = Circuit(self.n_qubits)
        for inst in instructions:
            t = inst.op.get_type()
            c.add_operation(t, inst.op.get_params(), inst.qubits)
        return c

    def instruction_to_unitary(self, instruction):
        t = instruction.op.get_type()
        if t in matrices_no_params:
            return matrices_no_params[t](instruction.qubits, self.n_qubits)
        elif t in matrices_with_params:
            return matrices_with_params[t](instruction.qubits, self.n_qubits, [p * math.pi for p in instruction.op.get_params()])
        else:
            raise ValueError("Unexpected instruction", instruction)

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
        if target in [control - 1, control + 1]:
            p = np.kron(np.eye(4 ** min(control, target)),
                           np.kron(self.cnot_noise @ self.get_adjacent_cnot_process_matrix(control, target),
                                   np.eye(4 ** (self.n_qubits - 1 - max(control, target)))))
            return p
        if target > control:
            g = self.basic_cnot_processes[1]
        else:
            g = self.basic_cnot_processes[-1]
        g = self.cnot_noise @ g
        d = abs(control - target)
        m = min(control, target)
        if d > 1:
            g = np.kron(np.eye(4 ** (d - 1)), g)
            g = np.moveaxis(g.reshape((4, 4) * (d + 1)), [0, d - 1, d + 1, 2 * d], [d - 1, 0, 2 * d, d + 1])\
                .reshape((4 ** (d + 1), 4 ** (d + 1)))
        return np.kron(np.eye(4 ** m), np.kron(g, np.eye(4 ** (self.n_qubits - 1 - max(control, target)))))

    def get_unitary_process_matrix(self, e):
        return np.vstack([[np.einsum('ij,ji->', e @ d2 @ e.transpose().conjugate(), d1, optimize=True) /
                           (2 ** self.n_qubits) for d1 in self.u_basis] for d2 in self.u_basis]).transpose()

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
        if s in self.process_cache:
            return self.process_cache[s]
        t = instruction.op.get_type()
        qubit = instruction.qubits[0]
        if t in matrices_no_params:
            gate = matrices_no_params[t]([0], 1)
        elif t in matrices_with_params:
            gate = matrices_with_params[t]([0], 1, [p * math.pi for p in instruction.op.get_params()])
        else:
            raise ValueError("Unexpected instruction", instruction)
        little_process = self.noise_process @ \
            np.vstack([[np.einsum('ij,ji->', gate @ d2 @ gate.transpose().conjugate(), d1, optimize=True) / 2
                        for d1 in one_qubit_diracs] for d2 in one_qubit_diracs]).transpose()
        z = np.kron(np.kron(np.eye(4 ** qubit), little_process), np.eye(4 ** (self.n_qubits - qubit - 1)))
        self.process_cache.update({s: z})
        return z

    def get_circuit_process_matrix(self, instructions):
        s = "".join([str(inst) for inst in instructions])
        if s in self.process_cache:
            return self.process_cache[s]
        l = len(instructions)
        for i in range(len(instructions)):
            end = "".join([str(inst) for inst in instructions[i:]])
            if end in self.process_cache:
                m = self.process_cache[end] @ self.get_circuit_process_matrix(instructions[:i])
                if l < 5:
                    self.process_cache.update({s: m})
                return m
            beginning = "".join([str(inst) for inst in instructions[:(l - i)]])
            if beginning in self.process_cache:
                m = self.get_circuit_process_matrix(instructions[(l - i):]) @ self.process_cache[beginning]
                if l < 5:
                    self.process_cache.update({s: m})
                return m
        m = np.eye(self.d2)
        for inst in instructions:
            if inst.op.get_type() == OpType.CX:
                m = self.cnot_processes[tuple(inst.qubits)] @ m
            else:
                m = self.get_single_qubit_gate_process_matrix(inst) @ m
        if l < 5:
            self.process_cache.update({s: m})
        return m

    def fidelity(self, instructions):
        s = np.einsum('kl,lk->', self.contracted, self.get_circuit_process_matrix(instructions), optimize=True).real
        return 2 ** (-3 * self.n_qubits) * s

    def reduce(self):
        if self.n_qubits < 6:
            for i in range(1, len(self.instructions) + 1):
                for s in {0, len(self.instructions) - i}:
                    self.process_cache.update({"".join([str(inst) for inst in self.instructions[s: s + i]]):
                                               self.get_circuit_process_matrix(self.instructions[s: s + i])})
        c = self.circuit.copy()
        cleanup.apply(c)
        self.original_fidelity = self.fidelity(c.get_commands())
        if self.verbose:
            print("original fidelity is", self.original_fidelity)
        applied = False
        while self.change_any_angle():  # or self.remove_any() or self.commute_any():
            applied = True
        if self.verbose:
            if applied:
                print("New fidelity is", self.fidelity(self.instructions))
            else:
                print("Didn't find anything to improve")
        if not applied:
            cleanup.apply(self.circuit)
            self.set_circuit(self.circuit)
        return self.fidelity(self.instructions)
