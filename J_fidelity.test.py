from J_fidelity import *
from noise import *
import math


print(f_pro(identity_channel(2), np.eye(4)))


theta = math.pi * 0.3
print(f_pro(identity_channel(2), np.kron(
    np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]), I1)))


p1 = 0.2
depolarising = depolarising_channel(p1)
print(f_pro(depolarising, I1))


gamma1 = 0.2
amplitude_damping = amplitude_damping_channel(gamma1)
print(f_pro(amplitude_damping, I1))


gamma2 = 0.2
phase_damping = phase_damping_channel(gamma2)
print(f_pro(phase_damping, I1))
