import numpy as np
import random
from typing import Dict
from metrics.J_fidelity import f_pro_experimental

ops = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
       "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
       "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
       "X": np.array([[0, 1], [1, 0]]),
       "I": np.eye(2)}


def run(length: int, unitary: np.ndarray, basic_gates: Dict[str, np.ndarray],
        min_fidelity: float = 0.9, max_iter: int = 1000) -> str:
    desc = "I" * length
    num_iter = 0
    l = list(basic_gates.keys())
    while f_pro_experimental(desc, unitary, key=basic_gates) < 1 - (1 - min_fidelity) * (1 - 1 / (num_iter + 1)) \
            and num_iter < max_iter:
        num_iter += 1
        options = [get_opt(desc, l) for _ in range(4)]
        desc = random.choices(options, weights=[f_pro_experimental(o, unitary, key=basic_gates) ** 21 for o in options])[0]
    return desc, f_pro_experimental(desc, unitary, key=basic_gates), num_iter


def get_opt(desc, reps):
    pos = random.randint(0, len(desc) - 1)
    replacement = random.choice(reps)
    if pos > 0:
        new = desc[:pos - 1] + replacement + desc[pos:]
    else:
        new = replacement + desc[1:]
    return new


#res = [run(10, random_unitary(), ops, max_iter=500) for _ in range(200)]
#print(res)
#print("average iterations taken", sum([r[2] for r in res]) / len(res))
#print("average fidelity", sum([r[1] for r in res]) / len(res))
#given_up = [r for r in res if r[2] == 500]
#if len(given_up) > 1:
#    print("average fidelity when giving up", sum([r[1] for r in given_up]) / len(given_up))
