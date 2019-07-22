from meet_in_the_middle import *
from Pauli import *
from common_gates import Rz


gate_set = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
            "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
            "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
            "X": np.array([[0, 1], [1, 0]])}

m = Rz(math.pi / 3, 0, 1)

print(m)

for n in range(10):
    a = approx(gate_set, m, 3 * n, 1 - n/10)
    print(a)
    if a is not None:
        print(distance(a[1], m))
    print()