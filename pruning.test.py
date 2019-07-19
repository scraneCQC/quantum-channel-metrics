from pruning_circuits import generate_random_circuit, prune_circuit
import J_fidelity
from noise import standard_noise_channels
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def run(n_qubits: int, circuit_length: int, tolerance: float, noise_strength: float, n_trials: int) \
        -> Tuple[float, float]:
    total = 0
    total_gates_cut = 0
    for _ in range(n_trials):
        circuit, key = generate_random_circuit(n_qubits, circuit_length)
        pruned = prune_circuit(circuit, tolerance)
        gates_cut = len(circuit) - len(pruned)
        total_gates_cut += gates_cut

        unitary = np.eye(2 ** n_qubits)
        for s in circuit[::-1]:
            unitary = key[s] @ unitary

        pruned_unitary = np.eye(2 ** n_qubits)
        for s in pruned[::-1]:
            pruned_unitary = key[s] @ pruned_unitary

        noise = standard_noise_channels(noise_strength, n_qubits)
        original_f = J_fidelity.f_pro_experimental(circuit, unitary, noise, key)
        pruned_f = J_fidelity.f_pro_experimental(pruned, unitary, noise, key)
        difference = pruned_f - original_f
        total += difference
    return total / n_trials, total_gates_cut / n_trials


def plot_tolerances():
    diffs = [run(3, 10, i/10, 0.01, 100)[0] for i in range(10)]

    best = max(list(range(10)), key=lambda i: diffs[i])
    print("The best tolerance threshold was", best / 10)

    plt.figure()
    plt.plot([i/10 for i in range(10)], diffs)

    plt.xlabel('Tolerance')
    plt.ylabel('Improvement in fidelity')
    plt.savefig('graphs/pruning_tolerance.png')


def plot_noise_strength():
    diffs = [run(3, 10, 0.1, 0.1 ** (i + 1), 100)[0] for i in range(10)]

    plt.figure()
    plt.semilogx([0.1 ** (i + 1) for i in range(10)], diffs)

    plt.xlabel('Noise strength')
    plt.ylabel('Improvement in fidelity')
    plt.savefig('graphs/pruning_noise.png')


plot_noise_strength()
