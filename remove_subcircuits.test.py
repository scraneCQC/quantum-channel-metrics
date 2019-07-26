from remove_subcircuits import *
import random
from density_runner import ops
from noise import standard_noise_channels
from pruning_circuits import generate_random_circuit


n_qubits = 1
noise = standard_noise_channels(1.889e-3, n_qubits)

# circuit1, key = generate_random_circuit(n_qubits, 40)
circuit1 = "".join(random.choices("SXHT", k=10))
key = ops

remover = SubcircuitRemover(circuit1, key, noise, n_qubits=n_qubits)
u1 = remover.unitary
print(circuit1)
print("The old circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit1, u1, noise, key))

remover.reduce_circuit()

circuit2 = remover.circuit
print(circuit2)
print("I reduced the gate count by", len(circuit1) - len(circuit2), "out of", len(circuit1))
u2 = remover.unitary
print("The new circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit2, u1, noise, key))

