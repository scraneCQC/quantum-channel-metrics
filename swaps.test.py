from spread_the_swaps import *
from J_distance import effect_of_noise
import random
from noise import *


noise_strength = 1e-2
qubit2_noise = single_qubit_depolarising_channel(noise_strength, 2, 3)


print(effect_of_noise([(0, 1) for _ in range(100)], [qubit2_noise], n_qubits=3, circuit_key=get_key(3)))
print(effect_of_noise(random.choices([(0, 1), (1, 2)], k=100), [qubit2_noise], n_qubits=3, circuit_key=get_key(3)))
