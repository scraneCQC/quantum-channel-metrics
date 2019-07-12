from diamond_distance import monte_carlo_f_algorithm
from noise import *
import math


print(monte_carlo_f_algorithm([np.kron(X, X)], identity_channel(2), 100))


print(monte_carlo_f_algorithm(not_channel(1), identity_channel(1), 100))


theta = math.pi * 0.3
print(monte_carlo_f_algorithm(identity_channel(2), [np.kron(
    np.array([[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]), I1)], 100))


p1 = 0.2
depolarising = depolarising_channel(p1)
print(monte_carlo_f_algorithm(depolarising, identity_channel(1), 100))


gamma1 = 0.2
amplitude_damping = amplitude_damping_channel(gamma1)
print(monte_carlo_f_algorithm(amplitude_damping, identity_channel(1), 100))


gamma2 = 0.2
phase_damping = phase_damping_channel(gamma2)
print(monte_carlo_f_algorithm(phase_damping, identity_channel(1), 100))