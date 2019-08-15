from pytket import Circuit, OpType
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from tket_pauli_gadgets.converter import converter
import math
import numpy as np
import scipy.optimize
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from cnot_decomposition import approximate_with_cnots, suggest_cnot_unitary
from common_gates import random_unitary
import cmath


np.set_printoptions(linewidth=1000)


def add_params_to_template(template: Circuit, params: np.ndarray):
    params = params.reshape((template.n_gates * 2 + template.n_qubits, 3))
    c = Circuit(template.n_qubits)
    for i in range(template.n_gates):
        cnot_gate = template.get_commands()[i]
        c.add_operation(OpType.U3, params[2 * i], [cnot_gate.qubits[0]])
        c.add_operation(OpType.U3, params[2 * i + 1], [cnot_gate.qubits[1]])
        c.CX(cnot_gate.qubits[0], cnot_gate.qubits[1])
    for q in range(template.n_qubits):
        c.add_operation(OpType.U3, params[template.n_gates * 2 + q], [q])
    return c

qft3 = Circuit(3)
qft3.CX(1, 0)
qft3.CX(1, 0)
qft3.CX(2, 0)
qft3.CX(2, 0)
qft3.CX(2, 1)
qft3.CX(2, 1)
qft3.CX(0, 1)
qft3.CX(0, 2)

def approximate_via_cnots(unitary):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    r = RewriteTket(Circuit(n_qubits), [], [])
    r.set_target_unitary(unitary)
    # cnot_circuit = approximate_with_cnots(unitary)
    cnot_circuit = qft3
    # print(dag_to_circuit(tk_to_dagcircuit(cnot_circuit)))
    n_params = (cnot_circuit.n_gates * 2 + n_qubits) * 3

    def distance(params):
        f = r.fidelity(add_params_to_template(cnot_circuit, params))
        return (1 - max(0, f) ** 0.5) ** 0.5

    best_distance = 1
    best_circ = None
    for _ in range(10):
        initial = np.random.random((n_params, ))
        res = scipy.optimize.minimize(distance, initial, method='SLSQP',
                                      options={'ftol': 0.01}, callback=lambda x: print(distance(x)))
        if res.success and res.fun < best_distance:
            best_distance = res.fun
            best_circ = add_params_to_template(cnot_circuit, res.x)
            if res.fun < 0.01:
                return best_circ, best_distance
    return best_circ, best_distance


converter.n_qubits = 3
#unitary = random_unitary(converter.n_qubits)
#unitary = np.eye(8)
#unitary[6][6] = 0
#unitary[6][7] = 1
#unitary[7][6] = 1
#unitary[7][7] = 0
#print(unitary)
#print(suggest_cnot_unitary(unitary))
n_qubits = 3
unitary = np.array([[cmath.exp(complex(0, i * j * 2 * math.pi / (2 ** n_qubits))) for j in range(2 ** n_qubits)] for i in range(2 ** n_qubits)]) / 2 ** (n_qubits / 2)
circuit, distance = approximate_via_cnots(unitary)
print(dag_to_circuit(tk_to_dagcircuit(circuit)))
print(distance)

