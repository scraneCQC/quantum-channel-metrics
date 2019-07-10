from S_fidelity import *
from noise import *
import math


print(f_min(identity_channel(2), identity_channel(2), 100))


theta = math.pi * 0.3
print(f_min(identity_channel(2), [np.kron(
    np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]), I1)], 100))


p1 = 0.2
depolarising = depolarising_channel(p1)
print(f_min(depolarising, identity_channel(1), 100))


gamma1 = 0.2
amplitude_damping = amplitude_damping_channel(gamma1)
print(f_min(amplitude_damping, identity_channel(1), 100))


gamma2 = 0.2
phase_damping = phase_damping_channel(gamma2)
print(f_min(phase_damping, identity_channel(1), 100))
