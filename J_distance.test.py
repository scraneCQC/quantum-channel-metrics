from J_distance import *
from noise import *
import math


print(j_distance([np.kron(X, X)], identity_channel(2)))


print(j_distance(not_channel(1), identity_channel(1)))


theta = math.pi * 0.3
print(j_distance(identity_channel(1), [np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]])]))


p1 = 0.2
depolarising = depolarising_channel(p1)
print(j_distance(depolarising, identity_channel(1)))


gamma1 = 0.2
amplitude_damping = amplitude_damping_channel(gamma1)
print(j_distance(amplitude_damping, identity_channel(1)))


gamma2 = 0.2
phase_damping = phase_damping_channel(gamma2)
print(j_distance(phase_damping, identity_channel(1)))
