from pytket import Circuit, OpType
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from tket_pauli_gadgets.converter import converter
import math
import numpy as np
from metrics.J_fidelity import bures
import scipy.optimize
from tket_pauli_gadgets.chem import get_circuit
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from common_gates import random_unitary


def build_circuit(n_qubits, n_layers, params=None):
    if params is None:
        params = np.zeros(3 * n_qubits * (n_layers + 1))
    params = params.reshape(((n_layers + 1), n_qubits, 3))
    c = Circuit(n_qubits)
    for i in range(n_layers):
        for q in range(n_qubits):
            c.add_operation(OpType.U3, params[i][q], [q])
        for q in range((n_qubits - 1) // 2):
            c.CX(2 * q + 1, 2 * q + 2)
        for q in range(n_qubits // 2):
            c.CX(2 * q, 2 * q + 1)
        for q in range((n_qubits - 1) // 2):
            c.CX(2 * q + 1, 2 * q + 2)
    for q in range(n_qubits):
        c.add_operation(OpType.U3, params[n_layers][q], [q])
    return c


def approximate(unitary, n_layers, fid_finder):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_params = (n_layers + 1) * n_qubits * 3
    initial = np.random.random((n_params, ))

    def bures_distance(params):
        f = fid_finder.fidelity(build_circuit(n_qubits, n_layers, params).get_commands())
        return (1 - max(0, f) ** 0.5) ** 0.5

    res = scipy.optimize.minimize(bures_distance, initial, method='Powell', options={'ftol': 1e-3, 'disp': True}, callback=lambda x: print(bures_distance(x)))
    if res.success:
        print("optimal params", res.x)
        print("distance", bures_distance(res.x))
        circ = build_circuit(n_qubits, n_layers, res.x)
        print("Circuit:\n", dag_to_circuit(tk_to_dagcircuit(circ)))
        return circ
    else:
        print(res)


converter.n_qubits = 4
circ = get_circuit([1.69252673e-07, 5.74166793e-02 + math.pi / 2], 13)
r = RewriteTket(circ, [], [])
r.set_circuit(approximate(r.target, 3, r))
r.reduce()
print("Reduced\n\n\n", dag_to_circuit(tk_to_dagcircuit(r.circuit)))
