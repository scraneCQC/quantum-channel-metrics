from tket_circuit_rewriter import RewriteTket
from noise import depolarising_channel
from pytket import Circuit
import random


def generate_random_circuit(n_qubits, length):
    c = Circuit(n_qubits)
    for _ in range(length):
        gate = random.randint(1, 10)
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


n_qubits = 4
circuit_length = 4
circuit = generate_random_circuit(n_qubits, circuit_length)

noise = depolarising_channel(2.227e-3, n_qubits)

rewriter = RewriteTket(circuit, [noise], verbose=True)
rewriter.reduce()
print(rewriter.circuit.get_commands())