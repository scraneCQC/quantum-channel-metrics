from remove_subcircuits import *
import random
from density_runner import ops
from noise import standard_noise_channels, depolarising_channel
from pruning_circuits import generate_random_circuit
from common_gates import clifford_T_gate_set, random_two_qubit_circuit, discrete_angle_key, get_cnot_key
import qft
from J_fidelity import f_pro_experimental, f_pro

random.seed(33)

n_qubits = 3
noise = [depolarising_channel(0.01, n_qubits)]

# circuit1, key = generate_random_circuit(n_qubits, 10)
# circuit1 = random.choices("SXHT", k=10)
# key = ops
# key = clifford_T_gate_set(n_qubits)
precision = 3
key = discrete_angle_key(precision, n_qubits)
key.update(get_cnot_key(n_qubits))
l = len(key)
weights = [1 for _ in range(3 * n_qubits * (2 ** precision - 1))] + [2 ** precision for _ in range(2 * (n_qubits - 1))]
circuit1 = random.choices(list(key.keys()), k=10, weights=weights)
# circuit1, key = qft.generate_circuit(n_qubits)
# circuit1, key = random_two_qubit_circuit()

remover = SubcircuitRemover(circuit1, key, noise, n_qubits=n_qubits, verbose=True)
u1 = remover.unitary
print(circuit1)
print("The old circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit1, u1, noise, key))

remover.reduce_circuit()

circuit2 = remover.circuit
print(circuit2)
print("I reduced the gate count by", len(circuit1) - len(circuit2), "out of", len(circuit1))
u2 = remover.unitary
print("The new circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit2, u1, noise, key))
print("The fidelity between the ideal circuits is", f_pro([u2], u1))
