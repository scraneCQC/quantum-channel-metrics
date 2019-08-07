from tket_pauli_gadgets.pauli_gadgets import random_pauli_gadget, pauli_gadget, random_gadget_circuit
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket, cleanup
from pytket import Transform, Circuit
from noise import phase_damping_channel
from tket_pauli_gadgets.noise_models import channels
import numpy as np
import matplotlib.pyplot as plt


single_noise, cnot_noise = channels(amplification=1)


def run(n_qubits):
    circuit = random_gadget_circuit(n_qubits, 3)
    rewriter = RewriteTket(circuit, single_noise, cnot_noise, verbose=True)
    rewriter.reduce()
    print(rewriter.circuit.get_commands())


def run_multiple(n_qubits, n_iter):
    c = Circuit(n_qubits)
    rewriter = RewriteTket(c, single_noise, cnot_noise, verbose=False)
    rewriter.verbose = True
    for _ in range(n_iter):
        circuit = random_pauli_gadget(n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        rewriter.reduce()
        print(rewriter.instructions)
        print("\n\n")


def run_multiple_angles(n_qubits, n_angles, s, noise1=[], noise2=[], rewriter=None):
    # Return average fidelity
    total = 0
    if rewriter is None:
        c = Circuit(n_qubits)
        rewriter = RewriteTket(c, noise1, noise2, verbose=False)
    for i in range(n_angles):
        circuit = pauli_gadget(i * 2 / n_angles, s, n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        f = rewriter.fidelity(rewriter.instructions)
        total = total + f
    return total / n_angles


def run_and_optimize_multiple_angles(n_qubits, n_angles, s, noise1=[], noise2=[], rewriter=None):
    # Return average fidelity
    total = 0
    if rewriter is None:
        c = Circuit(n_qubits)
        rewriter = RewriteTket(c, noise1, noise2, verbose=False)
    for i in range(n_angles):
        circuit = pauli_gadget(i * 2 / n_angles, s, n_qubits)
        Transform.OptimisePauliGadgets().apply(circuit)
        rewriter.set_circuit_and_target(circuit)
        rewriter.reduce()
        f = rewriter.fidelity(rewriter.instructions)
        total = total + f
    return total / n_angles


def plot_fidelity(s):
    noises = [i / 1000 for i in range(20)]
    fidelities = [run_multiple_angles(len(s), 20, s, [phase_damping_channel(p)], [phase_damping_channel(p, 2)]) for p in noises]
    improved_fidelities = [run_and_optimize_multiple_angles(len(s), 20, s, [phase_damping_channel(p)], [phase_damping_channel(p, 2)]) for p in noises]
    plt.figure()
    line_orig, = plt.plot(noises, fidelities)
    line_reduced, = plt.plot(noises, improved_fidelities)
    plt.xlabel("phase damping noise")
    plt.ylabel("average fidelity of "+s+" gadget")
    plt.legend((line_orig, line_reduced), ("original", "optimized"))
    plt.savefig("graphs/gadget_"+s+".png")
    plt.close()


def get_fid(s: str, angle: float, rewriter: RewriteTket):
    circuit = pauli_gadget(angle, s, len(s))
    cleanup.apply(circuit)
    rewriter.set_circuit_and_target(circuit)
    return rewriter.fidelity(rewriter.instructions)


def get_opt_fid(s: str, angle: float, rewriter: RewriteTket):
    rewriter.set_circuit_and_target(pauli_gadget(angle, s, len(s)))
    return rewriter.reduce()[1]


def plot_angles(s):
    rewriter = RewriteTket(Circuit(len(s)), single_noise, cnot_noise, verbose=False)
    angles = [2 * i / 100 for i in range(101)]
    fidelities = [get_fid(s, a, rewriter) for a in angles]
    opt_fidelities = [get_opt_fid(s, a, rewriter) for a in angles]
    plt.figure()
    line_orig, = plt.plot(angles, fidelities)
    line_reduced, = plt.plot(angles, opt_fidelities)
    plt.xlabel("alpha (multiples of pi)")
    plt.ylabel("fidelity of " + s + " gadget")
    plt.legend((line_orig, line_reduced), ("original", "optimized"), loc="upper left", bbox_to_anchor=(0.14, 0.95))
    plt.savefig("graphs/gadget_" + s + "_ibm_noise_model.png")
    plt.close()


np.set_printoptions(edgeitems=10, linewidth=1000)


plot_angles("XYZ")
