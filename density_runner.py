from noise import *


def apply_channel(channel, density):
    return sum([e @ density @ e.transpose().conjugate() for e in channel])


ops = {"S": np.array([[1, 0], [0, complex(0,1)]]),
       "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
       "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
       "X": np.array([[0, 1], [1, 0]]),
       "W": complex(0.5 ** 0.5, 0.5 ** 0.5) * np.eye(2),
       "I": np.eye(2)}


def run_by_matrices(string, start_density, p1=0, gamma1=0, gamma2=0):
    depolarising = depolarising_channel(p1)
    amplitude_damping = amplitude_damping_channel(gamma1)
    phase_damping = phase_damping_channel(gamma2)
    rho = start_density
    for s in string[::-1]:
        rho = apply_channel([ops[s]], rho)
        rho = apply_channel(depolarising, rho)
        rho = apply_channel(amplitude_damping, rho)
        rho = apply_channel(phase_damping, rho)
    return rho