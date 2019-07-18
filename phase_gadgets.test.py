import phase_gadgets
import math
from J_fidelity import effect_of_noise
from noise import standard_noise_channels


circuit1 = ["cnot01", "cnot12", "cnot23", "Rz3", "cnot23", "cnot12", "cnot01"]  # Original phase gadget
circuit2 = ["cnot01", "cnot32", "cnot12", "Rz2", "cnot12", "cnot32", "cnot01"]  # Balanced tree
circuit3 = ["cnot01-23", "cnot12", "Rz2", "cnot12", "cnot01-23"]  # Parallel CNOTs in balanced tree


theta = math.pi / 4

circuit_key = phase_gadgets.get_key(theta)


noise_strength = 0.001
noise_channels = standard_noise_channels(noise_strength, n_qubits=4)

print(effect_of_noise(circuit1, noise_channels, n_qubits=4, circuit_key=circuit_key))
print(effect_of_noise(circuit2, noise_channels, n_qubits=4, circuit_key=circuit_key))
print(effect_of_noise(circuit3, noise_channels, n_qubits=4, circuit_key=circuit_key))



