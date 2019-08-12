from metrics.S_fidelity import *
from noise import *
import math
from metrics.density_runner import ops
import matplotlib.pyplot as plt


ps = [x / 100 for x in range(101)]
ds = []
for p in ps:
    ds.append(bures(amplitude_damping_channel(p ** 2), [np.eye(2)]))
plt.plot(ps, ds)
plt.savefig("../graphs/amp distance")
quit()


print([np.trace(rho) for rho in random_densities(2, 10)])

print(experimental("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT",
                   np.array([[complex(math.cos(math.pi / 128), -math.sin(math.pi/128)), 0],
                             [0, complex(math.cos(math.pi / 128), math.sin(math.pi/128))]]), 100))


print(experimental("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT",
                   np.kron(np.array([[complex(math.cos(math.pi / 128), -math.sin(math.pi/128)), 0],
                             [0, complex(math.cos(math.pi / 128), math.sin(math.pi/128))]]), np.eye(2)), 100,
                   key={k: np.kron(v, np.eye(2)) for k, v in ops.items()}))


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
