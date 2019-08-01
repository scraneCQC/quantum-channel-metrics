from pauli_gadgets import random_pauli_gadget
from tket_circuit_rewriter import RewriteTket
from pytket import Transform, Circuit
from noise import standard_noise_channels

def run():
    n_qubits = 3
    circuit = random_pauli_gadget(n_qubits)
    Transform.OptimisePauliGadgets().apply(circuit)
    print("starting up")
    rewriter = RewriteTket(circuit, standard_noise_channels(0.01, n_qubits), verbose=True)
    print("let's go")
    rewriter.reduce()
    print(rewriter.instructions)


def run_multiple(n_qubits, n_iter):
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, standard_noise_channels(0.01, n_qubits), verbose=True)
    for _ in range(n_iter):
        circuit = random_pauli_gadget(n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        rewriter.reduce()
        print(rewriter.instructions)
        print("\n\n")


run_multiple(3, 10)