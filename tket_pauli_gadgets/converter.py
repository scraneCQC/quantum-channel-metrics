from pytket import Circuit, OpType
from common_gates import multi_qubit_matrix, H, S, V, cnot, Rx, Ry, Rz, U1, U3, X, Y, Z
import numpy as np
from functools import reduce
import math

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


class Converter:

    def __init__(self, n_qubits: int = 1):
        self.n_qubits = n_qubits

    def matrix_list_product(self, matrices, default_size=None):
        if len(matrices) == 0:
            if default_size is None:
                default_size = 2 ** self.n_qubits
            return np.eye(default_size)
        return reduce(lambda x, y: x @ y, matrices, np.eye(matrices[0].shape[0]))

    def instruction_to_unitary(self, instruction):
        t = instruction.op.get_type()
        if t in matrices_no_params:
            return matrices_no_params[t](instruction.qubits, self.n_qubits)
        elif t in matrices_with_params:
            return matrices_with_params[t](instruction.qubits, self.n_qubits,
                                           [p * math.pi for p in instruction.op.get_params()])
        else:
            raise ValueError("Unexpected instruction", instruction)

    def circuit_to_unitary(self, circuit):
        return self.matrix_list_product([self.instruction_to_unitary(i) for i in circuit.get_commands()[::-1]])

    def instructions_to_circuit(self, instructions):
        c = Circuit(self.n_qubits)
        for inst in instructions:
            t = inst.op.get_type()
            c.add_operation(t, inst.op.get_params(), inst.qubits)
        return c


converter = Converter()
