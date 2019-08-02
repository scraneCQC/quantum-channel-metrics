from tket_circuit_rewriter import RewriteTket
from noise import depolarising_channel
from pytket import Circuit, OpType
import random
import numpy as np
import cmath


def generate_random_circuit(n_qubits, length):
    c = Circuit(n_qubits)
    for _ in range(length):
        gate = random.randint(1, 7)
        if gate == 1:
            c.H(random.randint(0, n_qubits - 1))
        elif gate == 2:
            c.X(random.randint(0, n_qubits - 1))
        elif gate == 3:
            c.Y(random.randint(0, n_qubits - 1))
        elif gate == 4:
            c.Z(random.randint(0, n_qubits - 1))
        elif gate == 5:
            c.Rx(random.randint(0, n_qubits - 1), random.random() * 2)
        elif gate == 6:
            c.Ry(random.randint(0, n_qubits - 1), random.random() * 2)
        elif gate == 7:
            c.Rz(random.randint(0, n_qubits - 1), random.random() * 2)
        else:
            control = random.randint(0, n_qubits - 1)
            target = random.choice([x for x in range(n_qubits) if x in {control + 1, control - 1}])
            c.CX(target, control)
    return c


n_qubits = 3
circuit_length = 4
#circuit = generate_random_circuit(n_qubits, circuit_length)
circuit = Circuit(n_qubits)
circuit.add_operation(OpType.U3, [random.random() * 2 for _ in range(3)], [0])

noise = depolarising_channel(2.227e-3)

rewriter = RewriteTket(circuit, [noise], [], verbose=True)
np.set_printoptions(edgeitems=10, linewidth=1000)
old = rewriter.old_cnot_process(0, 2)
new = rewriter.get_cnot_process_matrix(0, 2)
print(all([all([cmath.isclose(old[i][j], new[i][j]) for i in range(64)]) for j in range(64)]))
"""
for i in range(circuit.n_gates):
    print(circuit.get_commands()[i])
    z1 = (rewriter.get_single_qubit_gate_process_matrix(circuit.get_commands()[i]))
    z2 = (rewriter.get_unitary_process_matrix(rewriter.instruction_to_unitary(circuit.get_commands()[i])))
    print(all([all([cmath.isclose(z1[i][j], z2[i][j])]) for j in range(16)]))
#rewriter.reduce()
#print(rewriter.circuit.get_commands())
"""