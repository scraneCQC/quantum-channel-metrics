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
from common_gates import random_unitary, random_two_qubit_circuit
from tket_pauli_gadgets.noise_models import channels
from itertools import product
import random
from noise import depolarising_channel


def build_circuit(n_qubits, n_layers, params=None):
    if params is None:
        params = np.zeros(3 * n_layers)
    params = params.reshape((n_layers, 3))
    c = Circuit(n_qubits)
    cnot_pairs = [(a, b) for a in range(n_qubits) for b in range(n_qubits) if a != b]
    for i in range(n_layers):
        control, target = cnot_pairs[i % len(cnot_pairs)]
        c.add_operation(OpType.U3, params[i], [control])
        c.CX(control, target)
    return c


def add_layer(circuit):
    n_qubits = circuit.n_qubits
    params = np.random.random((n_qubits, 3))
    for q in range(n_qubits):
        circuit.add_operation(OpType.U3, params[q], [q])
    for q in range(n_qubits // 2):
        circuit.CX(2 * q + 1, (2 * q + 2) % n_qubits)
    for q in range(n_qubits // 2):
        circuit.CX(2 * q + 1, 2 * q)


def build_two_qubit(params):
    params = params.reshape((7,3))
    c = Circuit(2)
    c.add_operation(OpType.U3, params[0], [0])
    c.add_operation(OpType.U3, params[1], [1])
    c.CX(0, 1)
    c.add_operation(OpType.U3, params[2], [0])
    c.add_operation(OpType.U3, params[3], [1])
    c.CX(1, 0)
    c.add_operation(OpType.U3, params[4], [0])
    c.add_operation(OpType.U3, params[5], [1])
    c.CX(0, 1)
    return c


def approximate(unitary, n_qubits, n_layers):
    c = build_circuit(n_qubits, n_layers, np.random.random(((n_layers + 1) * n_qubits * 3, )))
    # c = build_two_qubit(np.random.random((7, 3)))
    print(c.n_gates)
    n1, n2 = channels(amplification=100)
    r = RewriteTket(c, n1, n2)
    # print(dag_to_circuit(tk_to_dagcircuit(r.circuit)))
    r.set_target_unitary(unitary)
    indices = list(product(list(range(c.n_gates)), [0, 1, 2]))
    for p in range(1, 8):
        start = -1
        finish = 1
        num_iter = 0
        while finish - start > 0.01:
            random.shuffle(indices)
            num_iter = num_iter + 1
            print(num_iter)
            start = r.original_fidelity
            for i, j in indices:
                r.find_best_angle(i, j, p, adjust=True)
            finish = r.original_fidelity
        print("done", p, num_iter, finish)
    r.verbose = True
    r.reduce()
    r.verbose = False
    return r.circuit

# converter.n_qubits = 2
# d, k = random_two_qubit_circuit()
# unitary = converter.matrix_list_product([k[i] for i in d])
# a = approximate(unitary, 0, 0)
#
# quit()

#converter.n_qubits = 4
#circ = get_circuit([1.2, 0.3], 13)
#u = converter.circuit_to_unitary(circ)
#a = approximate(u, 4, 4)
#while a is False:
#    a = approximate(u, 4, 3)
#print(a.n_gates)
#print(dag_to_circuit(tk_to_dagcircuit(a)))


def approximate_scipy(unitary, n_layers, fid_finder):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_params = n_layers * 3
    initial = np.random.random((n_params, ))
    # initial = np.zeros((n_params, ))

    def distance(params):
        f = fid_finder.fidelity(build_circuit(n_qubits, n_layers, params).get_commands())
        return (1 - max(0, f) ** 0.5) ** 0.5

    res = scipy.optimize.minimize(distance, initial, method='Powell', jac=False, tol=0.01,
                                  options={'ftol': 0.01, 'disp': True}, callback=lambda x: print(distance(x)))
    if res.success:
        print("optimal params", res.x)
        print("distance", distance(res.x))
        circ = build_circuit(n_qubits, n_layers, res.x)
        #print("Circuit:\n", dag_to_circuit(tk_to_dagcircuit(circ)))
        return circ
    else:
        print(res)


converter.n_qubits = 4
circ = get_circuit((0.2, 0.7), 13)
r = RewriteTket(circ, [], [])
r.set_circuit(approximate_scipy(r.target, 48, r))
r.reduce()
#print("Reduced\n\n\n", dag_to_circuit(tk_to_dagcircuit(r.circuit)))
