import random
import math
from common_gates import adjacent_cnot, Rx, Ry, Rz
from typing import Iterable, Tuple, Dict, List, Any
import numpy as np
from metrics import J_fidelity
from noise import standard_noise_channels


def generate_random_circuit(n_qubits: int, n_gates: int) -> Tuple[List[str], Dict[str, np.ndarray]]:
    desc = []
    key = {}
    for _ in range(n_gates):
        if n_qubits == 1:
            gate_type = random.choice(["rx", "ry", "rz"])
        else:
            gate_type = random.choice(["cnot", "rx", "ry", "rz"])
        if gate_type == "cnot":
            direction = random.randint(0, 1)
            i = random.choice(list(range(n_qubits - 1)))
            if direction == 0:
                gate = adjacent_cnot(i, i + 1, n_qubits)
                s = "cnot" + str(i) + "-" + str(i + 1)
            else:
                gate = adjacent_cnot(i + 1, i, n_qubits)
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
    return desc, key


def prune_circuit(desc: Iterable, tolerance: float) -> Iterable:
    return [s for s in desc if not (s[0] == "r" and float(s.split("-", 1)[1]) < tolerance)]


def is_it_worth_it(unitary: np.ndarray, noise_strength: float) -> bool:
    n_qubits = int(math.log(unitary.shape[0], 2))
    noise = standard_noise_channels(noise_strength, n_qubits)
    circuit = ["A"]
    key = {"A": unitary}
    do_it = J_fidelity.f_pro_experimental(circuit, unitary, noise, key)
    dont_do_it = J_fidelity.f_pro([unitary], np.eye(unitary.shape[0]))
    if do_it > dont_do_it:
        return True
    return False


def prune_circuit_v2(desc: Iterable, key: Dict[Any, np.ndarray], noise_strength: float) -> Iterable:
    return [s for s in desc if is_it_worth_it(key[s], noise_strength)]


def find_threshold(noise_strength):
    min_angle = 0
    max_angle = 1
    for _ in range(100):
        mid_angle = (min_angle + max_angle) / 2
        if is_it_worth_it(Rz(mid_angle, 0, 1), noise_strength):
            max_angle = mid_angle
        else:
            min_angle = mid_angle
    return min_angle
