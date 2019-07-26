from remove_subcircuits import *
import random
from density_runner import ops
from noise import standard_noise_channels

noise = standard_noise_channels(0.001)
remover = SubcircuitRemover("".join(random.choices("STXH", k=100)), ops, noise)
circuit1 = remover.circuit
u1 = remover.unitary
print("The old circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit1, u1, noise, ops))

remover.reduce_circuit()

circuit2 = remover.circuit
u2 = remover.unitary
print("The new circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit2, u1, noise, ops))

