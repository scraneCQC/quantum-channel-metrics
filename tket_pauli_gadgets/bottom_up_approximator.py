from pytket import Circuit, OpType
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from tket_pauli_gadgets.converter import converter
import math
import numpy as np
import scipy.optimize
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from tket_pauli_gadgets.process_matrix import ProcessMatrixFinder
from common_gates import random_unitary, U3_params, U3_derivative
import random
import cmath


n_qubits = 4
n_layers = 16
converter.n_qubits = n_qubits
proc_finder = ProcessMatrixFinder(n_qubits, [], [])
unitary = random_unitary(n_qubits)
#qft
#unitary = np.array([[cmath.exp(complex(0, i * j * 2 * math.pi / (2 ** n_qubits))) for j in range(2 ** n_qubits)] for i in range(2 ** n_qubits)]) / 2 ** (n_qubits / 2)


def build_alternating_cnots_circuit(n_qubits, n_layers, params, skips=None):
    if skips is None:
        skips = []
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


def alternating_cnots_process_matrix(n_qubits, n_layers, params, skips = None):
    if skips is None:
        skips = []
    m = np.eye(4 ** n_qubits)
    cnot_count = 0
    cnots_full = 0
    for i in range(n_layers):
        if i % 2:
            single_layer = np.eye(1)
            cnot_layer = np.eye(1)
            r = n_qubits // 2
        else:
            single_layer = np.eye(4)
            cnot_layer = np.eye(4)
            r = (n_qubits - 1) // 2
        for q in range(r):
            if cnots_full not in skips:
                single_layer = np.kron(single_layer,
                                       proc_finder.unitary_to_process_matrix(U3_params(params[2 * cnot_count]), 1))
                single_layer = np.kron(single_layer,
                                       proc_finder.unitary_to_process_matrix(U3_params(params[2 * cnot_count + 1]), 1))
                cnot_layer = np.kron(cnot_layer, proc_finder.basic_cnot_processes[1])
                cnot_count += 1
            else:
                single_layer = np.kron(single_layer, np.eye(16))
                cnot_layer = np.kron(cnot_layer, np.eye(16))
            cnots_full += 1
        if single_layer.shape[0] < 4 ** n_qubits:
            single_layer = np.kron(single_layer, np.eye(4))
            cnot_layer = np.kron(cnot_layer, np.eye(4))
        m = single_layer @ cnot_layer @ m
    final_layer = np.eye(1)
    for q in range(n_qubits):
        final_layer = np.kron(final_layer, proc_finder.unitary_to_process_matrix(U3_params(params[-q]), 1))
    m = final_layer @ m
    return m


def grad(n_qubits, n_layers, params, skips, gate_index, param_index):
    if skips is None:
        skips = []
    m = np.eye(4 ** n_qubits)
    cnot_count = 0
    cnots_full = 0
    for i in range(n_layers):
        if i % 2:
            single_layer = np.eye(1)
            cnot_layer = np.eye(1)
            r = n_qubits // 2
        else:
            single_layer = np.eye(4)
            cnot_layer = np.eye(4)
            r = (n_qubits - 1) // 2
        for q in range(r):
            if cnots_full not in skips:
                if gate_index == 2 * cnot_count:
                    single_layer = np.kron(single_layer, proc_finder.unitary_to_process_matrix(U3_derivative(params[2 * cnot_count], param_index), 1))
                else:
                    single_layer = np.kron(single_layer,
                                           proc_finder.unitary_to_process_matrix(U3_params(params[2 * cnot_count]), 1))
                if gate_index == 2 * cnot_count + 1:
                    single_layer = np.kron(single_layer, proc_finder.unitary_to_process_matrix(U3_derivative(params[2 * cnot_count + 1], param_index), 1))
                else:
                    single_layer = np.kron(single_layer,
                                           proc_finder.unitary_to_process_matrix(U3_params(params[2 * cnot_count + 1]), 1))
                cnot_layer = np.kron(cnot_layer, proc_finder.basic_cnot_processes[1])
                cnot_count += 1
            else:
                single_layer = np.kron(single_layer, np.eye(16))
                cnot_layer = np.kron(cnot_layer, np.eye(16))
            cnots_full += 1
        if single_layer.shape[0] < 4 ** n_qubits:
            single_layer = np.kron(single_layer, np.eye(4))
            cnot_layer = np.kron(cnot_layer, np.eye(4))
        m = single_layer @ cnot_layer @ m
    final_layer = np.eye(1)
    for q in range(n_qubits):
        if gate_index == 2 * cnot_count:
            final_layer = np.kron(final_layer, proc_finder.unitary_to_process_matrix(
                U3_derivative(params[params.shape[0] - q - 1], param_index), 1))
        else:
            final_layer = np.kron(final_layer, proc_finder.unitary_to_process_matrix(U3_params(params[params.shape[0] - q - 1]), 1))
    m = final_layer @ m
    return m


def approximate_scipy(unitary, n_layers, fid_finder, skips=None):
    if skips is None:
        skips = []
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_cnots = (n_layers // 2) * (n_qubits // 2) + ((n_qubits - 1) // 2) * ((n_layers + 1) // 2)
    n_params = (n_cnots - len(skips)) * 6 + n_qubits * 3
    initial = np.random.random((n_params, ))

    def distance(params):
        #s = np.einsum('kl,lk->', fid_finder.contracted, alternating_cnots_process_matrix(n_qubits, n_layers,
        #            params.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips), optimize=True).real
        #f = 2 ** (-3 * n_qubits) * s
        f = fid_finder.fidelity(build_alternating_cnots_circuit(n_qubits, n_layers,
                        params.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips))
        return (1 - max(0, f) ** 0.5) ** 0.9  # Not actually Bures any more


    def jac(params):
        ds = [[np.einsum('kl,lk->', fid_finder.contracted, grad(n_qubits, n_layers,
                            params.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips, g, i),
                      optimize=True).real for i in range(3)] for g in range((n_cnots - len(skips)) * 2 + n_qubits)]
        f = 2 ** (-3 * n_qubits) * np.einsum('kl,lk->', fid_finder.contracted, alternating_cnots_process_matrix(n_qubits, n_layers,
                    params.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips), optimize=True).real
        return np.array(ds).flatten() * 2 ** (-3 * n_qubits) * (-0.25 / (f ** 0.5 - f) ** 0.5)

    print("initial", distance(initial))

    res = scipy.optimize.minimize(distance, initial, method='SLSQP', options={'ftol': 0.01}, callback=lambda x: print(distance(x)))
    if res.success:
        print("distance", res.fun)
        circ = build_alternating_cnots_circuit(n_qubits, n_layers, res.x.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips)
        return circ, res.fun
    else:
        print(res)
        return None, 1


def simulated_annealing(unitary, n_layers):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    n_cnots = (n_layers // 2) * (n_qubits // 2) + ((n_qubits - 1) // 2) * ((n_layers + 1) // 2)
    r = RewriteTket(Circuit(n_qubits), [], [])
    r.set_target_unitary(unitary)
    skips = []
    circuit, distance = approximate_scipy(unitary, n_layers, r, [])
    for k in range(10):
        new_skip = random.choice([i for i in range(n_cnots) if i not in skips])
        print("maybe I will skip", new_skip)
        new_circ, new_d = approximate_scipy(unitary, n_layers, r, skips + [new_skip])
        if new_d < distance:  # or random.random() < (1 - new_d + distance) / (5 * k + 1):
            print("skipping", new_skip)
            circuit, distance, skips = new_circ, new_d, skips + [new_skip]
        else:
            print("not skipping", new_skip)
        print(circuit.n_gates, circuit.n_gates_of_type(OpType.CX))
        if distance < 0.01:
            print("returning early", distance)
            return circuit, r.fidelity(circuit)
    print(distance)
    return circuit, r.fidelity(circuit)


circuit, fid = simulated_annealing(unitary, n_layers)
print("final fidelity", fid)
print(circuit.n_gates, circuit.n_gates_of_type(OpType.CX))
print(dag_to_circuit(tk_to_dagcircuit(circuit)))

