from meet_in_the_middle import *
from Pauli import *


gate_set = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
            "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
            "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
            "X": np.array([[0, 1], [1, 0]])}


print(approx(gate_set, np.eye(2), 6, 0.5))

