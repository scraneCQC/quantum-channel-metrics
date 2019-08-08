from pytket import Circuit, Transform
from Pauli import get_diracs
from functools import reduce
import numpy as np
import math
from tket_pauli_gadgets.process_matrix import ProcessMatrixFinder, matrices_with_params, matrices_no_params
from tket_pauli_gadgets import werner
from itertools import permutations

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
        self.target = target
        if target is None:
            self.target = self.circuit_to_unitary(circuit)
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

    @property
    def instructions(self):
        return self.circuit.get_commands()

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
        self.set_target_unitary(self.circuit_to_unitary(circuit))

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
                    res = dict()
                    for rounded in {round(params[i]), round(params[i] * 2) / 2}:
                        new_params = list(params)
                        new_params[i] = rounded
                        new_circuit = self.instructions_to_circuit(self.instructions[:index])
                        new_circuit.add_operation(inst.op.get_type(), new_params, inst.qubits)
                        new_circuit.add_circuit(self.instructions_to_circuit(self.instructions[index + 1:]), list(range(self.n_qubits)))
                        cleanup.apply(new_circuit)
                        new_fidelity = self.fidelity(new_circuit.get_commands())
                        if rounded in res:
                            if res[rounded][0] < new_fidelity - self.original_fidelity:
                                res[rounded] = (new_fidelity - self.original_fidelity, new_circuit, index, new_params)
                        else:
                            res[rounded] = (new_fidelity - self.original_fidelity, new_circuit, index, new_params)
                    best = max(res.values(), key=lambda x: x[0])
                    if best[0] > 0:
                        return best
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
        if f > 1e-7:
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

    def circuit_to_unitary(self, circuit):
        return self.matrix_list_product([self.instruction_to_unitary(i) for i in circuit.get_commands()[::-1]])

    def fidelity(self, instructions):
        mat = self.process_finder.instructions_to_process_matrix(instructions)
        if type(mat) != np.ndarray:
            #s = werner.einsum('kl,lk->', self.contracted, mat)
            mat = mat.todense()  # werner.einsum was failing here sometimes
        s = np.einsum('kl,lk->', self.contracted, mat, optimize=True).real
        f = 2 ** (-3 * self.n_qubits) * s
        if f < 0.3:  # Hack
            # TODO: better detection
            # TODO: check for arbitrary permutations
            if self.verbose:
                print("Assuming the tket optimization pass swapped qubits")
            for i in range(self.n_qubits):  # simple transposition
                for j in range(i):
                    mat2 = self.process_finder.get_swap_process_matrix(i, j) @ mat
                    s2 = np.einsum('kl,lk->', self.contracted, mat2, optimize=True).real
                    if s2 > s:
                        f = 2 ** (-3 * self.n_qubits) * s2
                    if f > 0.5:
                        if self.verbose:
                            print("assuming the swap was between", i, "and", j)
                        return f
            for i, j, k, l in [(0, 1, 2, 3), (0, 2, 1, 3), (0, 3, 1, 2)]: # double transposition
                mat2 = self.process_finder.get_swap_process_matrix(i, j) @ self.process_finder.get_swap_process_matrix(
                    l, k) @ mat
                s2 = np.einsum('kl,lk->', self.contracted, mat2, optimize=True).real
                if s2 > s:
                    f = 2 ** (-3 * self.n_qubits) * s2
                if f > 0.5:
                    if self.verbose:
                        print("assuming a double transposition of", i, j, "and", k, l)
                    return f
            for i in range(self.n_qubits):  # 3-cycles
                for j in range(i):
                    for k in range(i):
                        if j != k:
                            mat2 = self.process_finder.get_swap_process_matrix(i, j) @ self.process_finder.get_swap_process_matrix(j, k) @ mat
                            s2 = np.einsum('kl,lk->', self.contracted, mat2, optimize=True).real
                            if s2 > s:
                                f = 2 ** (-3 * self.n_qubits) * s2
                            if f > 0.5:
                                if self.verbose:
                                    print("assuming a 3-cycle between", i, j, "and", k)
                                return f
            for i, j, k, l in [(3, ) + p for p in permutations(list(range(3)))]:  # 4 - cycles
                mat2 = self.process_finder.get_swap_process_matrix(i, j) @ self.process_finder.get_swap_process_matrix(
                    j, k) @ self.process_finder.get_swap_process_matrix(k, l) @ mat
                s2 = np.einsum('kl,lk->', self.contracted, mat2, optimize=True).real
                if s2 > s:
                    f = 2 ** (-3 * self.n_qubits) * s2
                if f > 0.5:
                    if self.verbose:
                        print("assuming a 4-cycle of", i, j, k, l)
                    return f
            if self.verbose:
                print("didn't find a swap that corrected")
                print(instructions)
        return f

    def reduce(self):
        cleanup.apply(self.circuit)
        f = self.fidelity(self.circuit.get_commands())
        self.original_fidelity = f
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
        return (f, new_fidelity)


