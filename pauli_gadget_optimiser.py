from pauli_gadgets import random_pauli_gadget, pauli_gadget
from tket_circuit_rewriter import RewriteTket
from pytket import Transform, Circuit
from noise import channels, depolarising_channel, amplitude_damping_channel, phase_damping_channel
from itertools import product
import random
import matplotlib.pyplot as plt

one_qubit_noise = channels(0.01, 0.01, 0.01, 1)
cnot_noise = channels(0.02, 0.02, 0.02, 2)


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


def run_multiple_angles(n_qubits, n_angles, s, noise1, noise2):
    # Return average fidelity
    total = 0
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, noise1, noise2, verbose=False)
    for i in range(n_angles):
        circuit = pauli_gadget(i * 2 / n_angles, s, n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        total = total + rewriter.fidelity(rewriter.instructions)
    return total / n_angles


def run_and_optimize_multiple_angles(n_qubits, n_angles, s, noise1, noise2):
    # Return average fidelity
    total = 0
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, noise1, noise2, verbose=False)
    for i in range(n_angles):
        circuit = pauli_gadget(i * 2 / n_angles, s, n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        rewriter.reduce()
        total = total + rewriter.fidelity(rewriter.instructions)
    return total / n_angles


def plot_fidelity(s):
    noises = [i / 100 for i in range(20)]
    fidelities = [run_multiple_angles(len(s), 20, s, [depolarising_channel(p)], [depolarising_channel(p, 2)]) for p in noises]
    improved_fidelities = [run_and_optimize_multiple_angles(len(s), 20, s, [depolarising_channel(p)], [depolarising_channel(p, 2)]) for p in noises]
    plt.figure()
    line_orig, = plt.plot(noises, fidelities)
    line_reduced, = plt.plot(noises, improved_fidelities)
    plt.xlabel("depolarising noise")
    plt.ylabel("average fidelity of "+s+" gadget")
    plt.legend((line_orig, line_reduced), ("original", "optimized"))
    plt.savefig("graphs/gadget_"+s+".png")


plot_fidelity("ZZZ")

