import random
import math
from common_gates import cnot, Rx, Ry, Rz
from typing import Iterable, Tuple, Dict, List
import numpy as np


def generate_random_circuit(n_qubits: int, n_gates: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
    desc = []
    key = {}
    for _ in range(n_gates):
        gate_type = random.choice(["cnot", "rx", "ry", "rz"])
        if gate_type == "cnot":
            direction = random.randint(0, 1)
            i = random.choice(list(range(n_qubits - 1)))
            if direction == 0:
                gate = cnot(i, i + 1, n_qubits)
                s = "cnot" + str(i) + "-" + str(i + 1)
            else:
                gate = cnot(i + 1, i, n_qubits)
                s = "cnot" + str(i) + "-" + str(i + 1)
        elif gate_type == "rx":
            theta = random.random() * math.pi
            i = random.randint(0, n_qubits - 1)
            gate = Rx(theta, i, n_qubits)
            s = "rx" + str(i) + "-" + str(theta)
        elif gate_type == "ry":
            theta = random.random() * math.pi
            i = random.randint(0, n_qubits - 1)
            gate = Ry(theta, i, n_qubits)
            s = "ry" + str(i) + "-" + str(theta)
        elif gate_type == "rz":
            theta = random.random() * math.pi
            i = random.randint(0, n_qubits - 1)
            gate = Rz(theta, i, n_qubits)
            s = "rz" + str(i) + "-" + str(theta)
        desc.append(s)
        key.update({s: gate})
    return (desc, key)


def prune_circuit(desc: Iterable, tolerance: float) -> Iterable:
    return [s for s in desc if not (s[0] == "r" and float(s.split("-")[1]) < tolerance)]