import numpy as np
import random
import math
from typing import Dict, Iterable


class VPTree:
    def __init__(self, S):
        if len(S) == 1:
            self.left = None
            self.right = None
            self.p = S[0]
            self.mu = 0
        else:
            self.p = random.choice(S)
            distances = sorted([(r, distance(self.p[1], r[1])) for r in S if r[0] != self.p[0]], key=lambda x: x[1])
            mu_index = (len(distances) - 1) // 2
            self.mu = distances[mu_index][1]
            self.right = VPTree([x[0] for x in distances[mu_index:]])
            if mu_index == 0:
                self.left = None
                self.right = VPTree([x[0] for x in distances])
            else:
                self.left = VPTree([x[0] for x in distances[:mu_index]])


def distance(p, q):
    d = p.shape[0]
    return max((2 * d - 2 * np.trace(q.transpose().conjugate() @ p).real), 0) ** 0.5


def nearest_neighbour(unitary: np.ndarray, tree: VPTree, precision: float):
    # is there a member of the set within a distance precision of the given unitary?
    if tree is None:
        return False, None
    if tree.p is None:
        return False, None
    x = distance(tree.p[1], unitary)
    if x <= precision:
        return True, tree.p
    elif x <= tree.mu + precision:
        return nearest_neighbour(unitary, tree.left, precision)
    else:
        return nearest_neighbour(unitary, tree.right, precision)


def approx(gate_set: Dict[str, np.ndarray], unitary: np.ndarray, max_length: int, precision: float) -> Iterable:
    S = gate_set
    for i in range(math.ceil(max_length / 2)):
        print(i)
        new_S = {k1 + k2: v1 @ v2 for k1, v1 in gate_set.items() for k2, v2 in S.items()}
        tree = VPTree(list(S.items()))
        for k, v in S.items():
            success, w = nearest_neighbour(v.transpose().conjugate() @ unitary, tree, precision)
            if success:
                return k + w[0], v @ w[1]
        for k, v in new_S.items():
            success, w = nearest_neighbour(v.transpose().conjugate() @ unitary, tree, precision)
            if success:
                return k + w[0], v @ w[1]
        S = new_S