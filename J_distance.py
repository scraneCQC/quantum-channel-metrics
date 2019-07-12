import numpy as np
from scipy.linalg import sqrtm
from density_runner import apply_channel


def trace_norm(m1, m2):
    diff = complex(1, 0) * (m1 - m2)
    s = sqrtm(diff @ diff.transpose().conjugate())
    return np.trace(s).real / 2


def jamiolkowski(channel):
    d = channel[0].shape[0]
    big_channel = [np.kron(e, np.eye(d)) for e in channel]
    phi = sum([np.kron(c, c) for c in np.eye(d)])
    start_rho = np.outer(phi, phi) / d
    return apply_channel(big_channel, start_rho)


def j_distance(channel1, channel2):
    return trace_norm(jamiolkowski(channel1), (jamiolkowski(channel2)))

# "so far as we are aware, experimentally determining D_pro requires doing full process
# tomography, which for a d-dimensional quantum system
# requires the estimation of d ** 4 − d ** 2 observable averages"

