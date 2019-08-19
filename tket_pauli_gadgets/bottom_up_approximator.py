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


random.seed(42)

n_qubits = 3
n_layers = 6 * 4 ** (n_qubits - 1) // n_qubits
converter.n_qubits = n_qubits
proc_finder = ProcessMatrixFinder(n_qubits, [], [])
unitary = random_unitary(n_qubits)
#qft
#unitary = np.array([[cmath.exp(complex(0, i * j * 2 * math.pi / (2 ** n_qubits))) for j in range(2 ** n_qubits)] for i in range(2 ** n_qubits)]) / 2 ** (n_qubits / 2)
r = RewriteTket(Circuit(n_qubits), [], [])
r.set_target_unitary(unitary)


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
        f = fid_finder.fidelity(build_alternating_cnots_circuit(n_qubits, n_layers,
                        params.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips))
        if f > 0.99:
            return 0
        return (1 - max(0, f) ** 0.5) ** 0.9  # Not actually Bures any more

    print("initial", distance(initial))

    res = scipy.optimize.minimize(distance, initial, method='Powell', options={'ftol': 0.001},
                                  callback=lambda x: print(distance(x) ** 0.45))  # print Bures
    if res.success:
        print("distance", res.fun)
        circ = build_alternating_cnots_circuit(n_qubits, n_layers, res.x.reshape(((n_cnots - len(skips)) * 2 + n_qubits, 3)), skips)
        return circ, fid_finder.fidelity(circ)
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
        if new_d <= distance:  # or random.random() < (1 - new_d + distance) / (5 * k + 1):
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


circuit, fid = approximate_scipy(unitary, n_layers, r)
print("final fidelity", fid)
print(circuit.n_gates, circuit.n_gates_of_type(OpType.CX))
print(dag_to_circuit(tk_to_dagcircuit(circuit)))

