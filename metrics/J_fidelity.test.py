from metrics.J_fidelity import *
from noise import *
import math
from Pauli import *
from density_runner import ops


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


print(f_pro_experimental("I", np.eye(2)))
print(f_pro_simulated("I", np.eye(2)))
print()

print(f_pro_experimental("X", np.eye(2)))
print(f_pro_simulated("X", np.eye(2)))
print()


c = math.cos(math.pi/256)
s = math.sin(math.pi/256)
print(f_pro_experimental("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT",
                         np.array([[complex(c, -s), 0], [0, complex(c, s)]])))
print(f_pro_simulated("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT",
                         np.array([[complex(c, -s), 0], [0, complex(c, s)]])))
print(f_pro_simulated("HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT",
                         np.kron(np.array([[complex(c, -s), 0], [0, complex(c, s)]]), np.eye(2))))
print()


theta = 1
U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])


for name, mat in ops.items():
    print(name)
    print("actual ", f_pro_experimental(name, U))
    print("simulated ", f_pro_simulated(name, U))
    print()