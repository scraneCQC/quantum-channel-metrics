import numpy as np
from scipy.linalg import sqrtm


def trace_norm(m1, m2):
    diff = (m1 - m2)
    if np.isrealobj(diff):
        diff = diff * complex(0, 1)  # silly thing can't find square roots if it expects them to be real
    s = sqrtm(diff @ diff.transpose().conjugate())
    if np.isrealobj(diff):
        s = s * complex(0.5 ** 0.5, - 0.5 ** 0.5)  # this just cancels it out
    return np.trace(s).real / 2


def apply_channel(channel, density):
    return sum([e @ density @ e.transpose().conjugate() for e in channel])


def jamiolkowski(channel):
    d = channel[0].shape[0]
    big_channel = [np.kron(e, np.eye(d)) for e in channel]
    phi = sum([np.kron(c, c) for c in np.eye(d)])
    start_rho = np.outer(phi, phi) / d
    return apply_channel(big_channel, start_rho)


def j_distance(channel1, channel2):
    #print(jamiolkowski(channel1))
    return trace_norm(jamiolkowski(channel1), (jamiolkowski(channel2)))

