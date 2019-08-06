from pytket import Circuit, Transform
from Pauli import get_diracs
from functools import reduce
import numpy as np
import math
from tket_pauli_gadgets.process_matrix import ProcessMatrixFinder, matrices_with_params, matrices_no_params
from tket_pauli_gadgets import werner

# noinspection PyCallByClass
cleanup = Transform.sequence([
                              Transform.repeat(Transform.sequence([
                                  Transform.RemoveRedundancies(),
                                  Transform.ReduceSingles(),
                                  Transform.CommuteRzRxThroughCX()])),
                              Transform.OptimisePauliGadgets(),
                              Transform.ReduceSingles()])


class RewriteTket:

    def __init__(self, circuit: Circuit, noise_channels, cnot_noise_channels, target=None, verbose=False):
        self.verbose = verbose
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.use_sparse = self.n_qubits > 4
        self.instructions = circuit.get_commands()
        self.target = target
        if target is None:
            self.target = self.matrix_list_product([self.instruction_to_unitary(inst) for inst in self.instructions[::-1]])
        self.u_basis = get_diracs(self.n_qubits)
        self.state_basis = np.array(self.u_basis)
        if self.use_sparse:
            a = werner.einsum('knm,in->kim', self.state_basis, self.target)
            b = werner.einsum('kim,jm->kij', a, self.target.conjugate())
            self.contracted = werner.einsum('kij,lji->kl', b, self.state_basis)
        else:
            self.contracted = np.einsum('knm,lji,in,jm->kl',
                            self.state_basis, self.state_basis, self.target, self.target.conjugate(), optimize=True)
        self.process_finder = ProcessMatrixFinder(self.n_qubits, noise_channels, cnot_noise_channels)
        self.original_fidelity = -1
        if self.verbose:
            print("Ready to go")

    def set_circuit(self, circuit: Circuit):
        self.circuit = circuit
        self.n_qubits = circuit.n_qubits
        self.instructions = circuit.get_commands()

    def set_target_unitary(self, target: np.ndarray):
        self.target = target
        if self.use_sparse:
            a = werner.einsum('knm,in->kim', self.state_basis, self.target)
            b = werner.einsum('kim,jm->kij', a, self.target.conjugate())
            self.contracted = werner.einsum('kij,lji->kl', b, self.state_basis)
        else:
            self.contracted = np.einsum('knm,lji,in,jm->kl', self.state_basis, self.state_basis, self.target,
                                        self.target.conjugate(), optimize=True)

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

    def should_change_angle(self, index):
        inst = self.instructions[index]
        if inst.op.get_type() in matrices_with_params:
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
                diffs. append((self.should_change_angle(i)))
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

    def fidelity(self, instructions):
        mat = self.process_finder.instructions_to_process_matrix(instructions)
        if type(mat) != np.ndarray:
            s = werner.einsum('kl,lk->', self.contracted, mat).real
        else:
            s = np.einsum('kl,lk->', self.contracted, mat, optimize=True).real
        return 2 ** (-3 * self.n_qubits) * s

    def reduce(self):
        cleanup.apply(self.circuit)
        self.set_circuit(self.circuit)
        self.original_fidelity = self.fidelity(self.circuit.get_commands())
        if self.verbose:
            print("original fidelity is", self.original_fidelity)
        applied = False
        while self.change_any_angle():
            applied = True
        new_fidelity = self.fidelity(self.instructions)
        if self.verbose:
            if applied:
                print("New fidelity is", new_fidelity)
            else:
                print("Didn't find anything to improve")
        return new_fidelity


