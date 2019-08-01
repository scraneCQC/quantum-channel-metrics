from pauli_gadgets import random_pauli_gadget
from tket_circuit_rewriter import RewriteTket
from pytket import Transform
from noise import standard_noise_channels

def run():
    n_qubits = 3
    circuit = random_pauli_gadget(n_qubits)
    Transform.OptimisePauliGadgets().apply(circuit)
    print("starting up")
    rewriter = RewriteTket(circuit, standard_noise_channels(0.03, n_qubits), verbose=True)
    print("let's go")
    rewriter.reduce()
    print(rewriter.instructions)

run()