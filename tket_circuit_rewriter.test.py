from tket_circuit_rewriter import RewriteTket
from noise import depolarising_channel
from pytket import Circuit
import random
import numpy as np


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


n_qubits = 2
circuit_length = 4
#circuit = generate_random_circuit(n_qubits, circuit_length)
circuit = Circuit(2)
circuit.H(0)
circuit.X(0)
circuit.Y(0)
circuit.Z(0)

noise = depolarising_channel(2.227e-3, n_qubits)

rewriter = RewriteTket(circuit, [noise], verbose=True)
np.set_printoptions(edgeitems=10, linewidth=1000)
for i in range(circuit.n_gates):
    print(circuit.get_commands()[i])
    print(rewriter.get_single_qubit_process_matrix(circuit.get_commands()[i]))
    print(rewriter.get_unitary_process_matrix(rewriter.instruction_to_matrix(circuit.get_commands()[i])))
#rewriter.reduce()
#print(rewriter.circuit.get_commands())