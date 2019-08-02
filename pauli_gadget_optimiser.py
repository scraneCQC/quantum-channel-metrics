from pauli_gadgets import random_pauli_gadget, pauli_gadget
from tket_circuit_rewriter import RewriteTket
from pytket import Transform, Circuit
from noise import standard_noise_channels
from itertools import product
import random

one_qubit_noise = standard_noise_channels(1.617e-2)
cnot_noise = standard_noise_channels(1.735e-1, 2)


def run(n_qubits):
    circuit = random_pauli_gadget(n_qubits)
    Transform.OptimisePauliGadgets().apply(circuit)
    rewriter = RewriteTket(circuit, one_qubit_noise, cnot_noise, verbose=True)
    rewriter.reduce()
    print(rewriter.instructions)


def run_multiple(n_qubits, n_iter):
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, one_qubit_noise, cnot_noise, verbose=False)
    rewriter.verbose = True
    for _ in range(n_iter):
        circuit = random_pauli_gadget(n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        rewriter.reduce()
        print(rewriter.instructions)
        print("\n\n")


def run_multiple_angles(n_qubits, n_angles, s):
    # Return average gain in fidelity
    total = 0
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, one_qubit_noise, cnot_noise, verbose=False)
    for i in range(n_angles):
        circuit = pauli_gadget(i * 2 / n_angles, s, n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        total = total + rewriter.reduce()
    return total / n_angles


n_qubits = 5
# for s in ["".join(x) for x in product("XYZ", repeat=n_qubits)]:
#     print(s, run_multiple_angles(n_qubits, 10, s))
run_multiple(n_qubits, 10)