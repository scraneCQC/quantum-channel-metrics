from pytket import Circuit, OpType
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from tket_pauli_gadgets.converter import converter
import math
import numpy as np
import scipy.optimize
from tket_pauli_gadgets.chem import get_circuit
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from cnot_decomposition import approximate_with_cnots, suggest_cnot_unitary


def put_params_in_cnot_circuit(cnot_circ: Circuit, params):
    params = np.reshape(params, (cnot_circ.n_gates_of_type(OpType.CX) + cnot_circ.n_qubits, 2, 3))
    c = Circuit(cnot_circ.n_qubits)
    i = 0
    for com in cnot_circ.get_commands():
        if com.op.get_type() == OpType.CX:
            c.add_operation(OpType.U3, params[i][0], com.qubits[0])
            c.add_operation(OpType.U3, params[i][1], com.qubits[1])
            c.CX(com.qubits[0], com.qubits[1])
            i += 1
    for q in range(cnot_circ.n_qubits):
        c.add_operation(OpType.U3, params[i + q][0], [q])
    return c


def approximate_via_cnots(unitary):
    dim = unitary.shape[0]
    n_qubits = int(math.log(dim, 2))
    converter.n_qubits = n_qubits
    r = RewriteTket(Circuit(n_qubits), [], [])
    r.set_target_unitary(unitary)
    cnot_circuit = approximate_with_cnots(unitary)
    print(dag_to_circuit(tk_to_dagcircuit(cnot_circuit)))
    n_params = (cnot_circuit.n_gates_of_type(OpType.CX ) + n_qubits)* 6
    initial = np.zeros((n_params, ))

    def distance(params):
        f = r.fidelity(put_params_in_cnot_circuit(cnot_circuit, params).get_commands())
        return (1 - max(0, f) ** 0.5) ** 0.5

    res = scipy.optimize.minimize(distance, initial, method='Powell', jac=False,
                                  options={'ftol': 0.01}, callback=lambda x: print(distance(x)))
    if res.success:
        print("distance", res.fun)
        circ = put_params_in_cnot_circuit(cnot_circuit, res.x)
        return circ, res.fun
    else:
        print(res)


converter.n_qubits = 4
circ = get_circuit((0.5, 0.7), 13)
unitary = converter.circuit_to_unitary(circ)
print(suggest_cnot_unitary(unitary))
circuit, distance = approximate_via_cnots(unitary)
print(dag_to_circuit(tk_to_dagcircuit(circuit)))