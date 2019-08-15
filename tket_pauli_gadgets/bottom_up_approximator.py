from pytket import Circuit, OpType
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from tket_pauli_gadgets.converter import converter
import math
import numpy as np
import scipy.optimize
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from tket_pauli_gadgets.process_matrix import ProcessMatrixFinder
from common_gates import random_unitary
import random


def build_alternating_cnots_circuit(n_qubits, n_layers, params=None, skips=None):
    if skips is None:
        skips = []
    if params is None:
        params = np.zeros(((n_layers - len(skips)) * 2 + n_qubits, 3))
    else:
        params = params.reshape(((n_layers - len(skips)) * 2 + n_qubits, 3))
    c = Circuit(n_qubits)
    cnot_count = 0
    cnots_full = 0
    for i in range(n_layers):
        if i % 2:
            for q in range(n_qubits // 2):
                if cnots_full not in skips:
                    c.add_operation(OpType.U3, params[2 * cnot_count], [2 * q])
                    c.add_operation(OpType.U3, params[2 * cnot_count + 1], [2 * q + 1])
                    c.CX(2 * q, 2 * q + 1)
                    cnot_count += 1
                cnots_full += 1
        else:
            for q in range((n_qubits - 1) // 2):
                if cnots_full not in skips:
                    c.add_operation(OpType.U3, params[2 * cnot_count], [2 * q + 1])
                    c.add_operation(OpType.U3, params[2 * cnot_count + 1], [2 * q + 2])
                    c.CX(2 * q + 1, 2 * q + 2)
                    cnot_count += 1
                cnots_full += 1
    for q in range(n_qubits):
        c.add_operation(OpType.U3, params[-q], [q])
    return c


# process = ProcessMatrixFinder(4, [], [])
#
#
# def fidelity(n_qubits, n_layers, params, skips, re):
#     if params is None:
#         params = np.zeros((n_layers + 1, n_qubits, 3))
#     else:
#         params = params.reshape((n_layers + 1, n_qubits, 3))
#     process_matrix = np.eye(4 ** n_qubits)
#     cnot_count = 0
#     for i in range(n_layers):
#         for q in range(n_qubits):
#             m = process.unitary_to_process_matrix(U3_params(params[0][q]), 1)
#             m = np.kron(np.eye(4 ** q), np.kron(m, np.eye(4 ** (n_qubits - q - 1))))
#             process_matrix = m @ process_matrix
#         if i % 2:
#             for q in range(n_qubits // 2):
#                 if cnot_count not in skips:
#                     process_matrix = process.get_cnot_process_matrix(2 * q, 2 * q + 1)
#                 cnot_count += 1
#         else:
#             for q in range((n_qubits - 1) // 2):
#                 if cnot_count not in skips:
#                     process_matrix = process.get_cnot_process_matrix(2 * q + 1, 2 * q + 2)
#                 cnot_count += 1
#     for q in range(n_qubits):
#         m = process.unitary_to_process_matrix(U3_params(params[n_layers][q]), 1)
#         m = np.kron(np.eye(4 ** q), np.kron(m, np.eye(4 ** (n_qubits - q - 1))))
#         process_matrix = m @ process_matrix
#     s = np.einsum('kl,lk->', re.contracted, m, optimize=True).real
#     f = 2 ** (-3 * n_qubits) * s
#     return f


def approximate_scipy(unitary, n_cnots, fid_finder, skips=None):
    if skips is None:
        skips = []
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_params = (n_cnots - len(skips)) * 6 + n_qubits * 3
    initial = np.random.random((n_params, ))
    # initial = np.zeros((n_params, ))

    def distance(params):
        f = fid_finder.fidelity(build_alternating_cnots_circuit(n_qubits, n_cnots, params, skips))
        #if f > 0.99:
        #    return 0
        return (1 - max(0, f) ** 0.5) ** 0.5

    print("initial", distance(initial))

    res = scipy.optimize.minimize(distance, initial, method='Powell', jac=False,
                                  options={'ftol': 0.1}, callback=lambda x: print(distance(x)))
    if res.success:
        print("distance", res.fun)
        circ = build_alternating_cnots_circuit(n_qubits, n_cnots, res.x, skips)
        return circ, res.fun
    else:
        print(res)


def simulated_annealing(unitary, n_layers):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_cnots = (n_layers // 2) * (n_qubits // 2) + ((n_qubits - 1) // 2) * ((n_layers + 1) // 2)
    r = RewriteTket(Circuit(n_qubits), [], [])
    r.set_target_unitary(unitary)
    skips = []
    circuit, distance = approximate_scipy(unitary, n_cnots, r, [])
    for k in range(10):
        new_skip = random.choice([i for i in range(n_cnots) if i not in skips])
        print("maybe I will skip", new_skip)
        new_circ, new_d = approximate_scipy(unitary, n_cnots, r, skips + [new_skip])
        if new_d < distance:  # or random.random() < (1 - new_d + distance) / (5 * k + 1):
            print("skipping", new_skip)
            circuit, distance, skips = new_circ, new_d, skips + [new_skip]
        else:
            print("not skipping", new_skip)
        print(circuit.n_gates, circuit.n_gates_of_type(OpType.CX))
        if distance < 0.01:
            print("returning early")
            return circuit, r.fidelity(circuit.get_commands())
    return circuit, distance


converter.n_qubits = 3
# unitary = random_unitary(3)

# Toffoli
unitary = np.eye(8)
unitary[6][6] = 0
unitary[6][7] = 1
unitary[7][6] = 1
unitary[7][7] = 0

circuit, distance = simulated_annealing(unitary, 16)
print(circuit.n_gates, circuit.n_gates_of_type(OpType.CX))
print(dag_to_circuit(tk_to_dagcircuit(circuit)))

