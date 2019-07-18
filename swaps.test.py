from spread_the_swaps import *
from diamond_distance import effect_of_noise
import random


noise_strength = 1e-4
print(effect_of_noise([(0, 1) for _ in range(20)], p1=noise_strength, gamma1=noise_strength,
                      gamma2=noise_strength, n_qubits=3, circuit_key=get_key(3)))
print(effect_of_noise(random.choices([(0, 1), (1, 2)], k=20), p1=noise_strength, gamma1=noise_strength,
                      gamma2=noise_strength, n_qubits=3, circuit_key=get_key(3)))
