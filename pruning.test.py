from pruning_circuits import generate_random_circuit, prune_circuit
import J_fidelity
from noise import standard_noise_channels
import numpy as np


circuit, key = generate_random_circuit(3, 10)
print(circuit)
pruned = prune_circuit(circuit, 0.2)
print(pruned)
print("Pruning removed", len(circuit) - len(pruned), "gates")


unitary = np.eye(8)
for s in circuit[::-1]:
    unitary = key[s] @ unitary


noise = standard_noise_channels(0.01, 3)
print(J_fidelity.f_pro_experimental(circuit, unitary, noise, key))
print(J_fidelity.f_pro_experimental(pruned, unitary, noise, key))