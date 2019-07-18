from spread_the_swaps import *
from S_fidelity import effect_of_noise
import random
from noise import *


noise_strength = 1e-3
qubit2_noise = single_qubit_depolarising_channel(noise_strength, 2, 3)
qubit1_noise = single_qubit_depolarising_channel(noise_strength, 1, 3)


my_circuit = [(0, 1) for _ in range(100)]


print(effect_of_noise(my_circuit, [qubit1_noise], n_qubits=3, circuit_key=get_key(3)))
print(effect_of_noise(my_circuit, [qubit2_noise], n_qubits=3, circuit_key=get_key(3)))
print(effect_of_noise(my_circuit, [qubit1_noise, qubit2_noise], n_qubits=3, circuit_key=get_key(3)))
