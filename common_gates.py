import numpy as np
import math
import cmath
import random
from typing import Dict
from scipy.linalg import block_diag
from Pauli import X, Y, Z
from functools import reduce
from scipy.stats import unitary_group


cnot12 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cnot21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def adjacent_cnot(i: int, j: int, n_qubits: int) -> np.ndarray:
    if j == i + 1:
        gate = cnot12
    elif j == i - 1:
        gate = cnot21
    else:
        raise NotImplementedError("can only do CNOTs between adjacent qubits right now")
    m = min(i, j)
    for _ in range(m):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - m - 2):
        gate = np.kron(gate, np.eye(2))
    return gate


def cnot(control: int, target: int, n_qubits: int) -> np.ndarray:
    if control == target:
        raise ValueError("control and target cannot be equal")
    elif control < target:
        g = cnot12
    else:
        g = cnot21
    d = abs(control - target)
    m = min(control, target)
    ma = max(control, target)
    if d > 1:
        g = np.kron(np.eye(2 ** (d - 1)), g)
        g = np.moveaxis(g.reshape((2,2) * (d + 1)), [0, d - 1, d + 1, 2 * d], [d - 1, 0, 2 * d, d + 1]).reshape((2 ** (d + 1), 2 ** (d + 1)))
    return np.kron(np.eye(2 ** m), np.kron(g, np.eye(2 ** (n_qubits - 1 - ma))))


cnot_key1 = {"cnot" + str(i) + str(i + 1): adjacent_cnot(i, i + 1, 4) for i in range(3)}
cnot_key2 = {"cnot" + str(i + 1) + str(i): adjacent_cnot(i + 1, i, 4) for i in range(3)}
cnot_key = {**cnot_key1, **cnot_key2, "cnot01-23": np.kron(cnot12, cnot12), "cnot01-32": np.kron(cnot12, cnot21)}


def get_cnot_key(n_qubits):
    key1 = {"cnot" + str(i) + str(i + 1): adjacent_cnot(i, i + 1, n_qubits) for i in range(n_qubits - 1)}
    key2 = {"cnot" + str(i + 1) + str(i): adjacent_cnot(i + 1, i, n_qubits) for i in range(n_qubits - 1)}
    return {**key1, **key2}


swap_basic = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])

def swap(i: int, j: int, n_qubits: int):
    g = swap_basic
    d = abs(i - j)
    m = min(i, j)
    ma = max(i, j)
    if d > 1:
        g = np.kron(np.eye(2 ** (d - 1)), g)
        g = np.moveaxis(g.reshape((2,2) * (d + 1)), [0, d - 1, d + 1, 2 * d], [d - 1, 0, 2 * d, d + 1]).reshape((2 ** (d + 1), 2 ** (d + 1)))
    return np.kron(np.eye(2 ** m), np.kron(g, np.eye(2 ** (n_qubits - 1 - ma))))


def Rz(angle: float, i: int, n_qubits: int) -> np.ndarray:
    gate = np.array([[cmath.exp(complex(0, - angle / 2)), 0], [0, cmath.exp(complex(0, angle / 2))]])
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def get_Rz_key(angle: float, n_qubits: int) -> Dict[str, np.ndarray]:
    return {"Rz" + str(i): Rz(angle, i, n_qubits) for i in range(4)}


def CRz(angle: float, control: int, target: int, n_qubits: int) -> np.ndarray:
    if target == control:
        raise ValueError("control and target must be different")
    if target < control:
        gate = block_diag(np.eye(2 ** (n_qubits - control - 1)), Rz(angle, target - control - 1, n_qubits - control - 1))
        for _ in range(control):
            gate = np.kron(np.eye(2), gate)
        return gate
    else:
        raise NotImplementedError("sorry control must be lower than target")


def U1(theta: float, i: int, n_qubits: int) -> np.ndarray:
    gate = np.array([[1, 0], [0, cmath.exp(complex(0, theta))]])
    return multi_qubit_matrix(gate, i, n_qubits)


def U3(theta: float, phi, lam, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e1 = cmath.exp(complex(0, phi))
    e2 = cmath.exp(complex(0, lam))
    gate = np.array([[c, -e2 * s], [e1 * s, e1 * e2 * c]])
    return multi_qubit_matrix(gate, i, n_qubits)
    #gate = Rz(phi, 0, 1) @ Ry(theta, 0, 1) @ Rz(lam, 0, 1)
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def U3_params(params):
    return U3(params[0], params[1], params[2], 0, 1)


def U3_derivative(params, i):
    theta = params[0]
    phi = params[1]
    lam = params[2]
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e1 = cmath.exp(complex(0, phi))
    e2 = cmath.exp(complex(0, lam))
    if i == 0:
        return np.array([[-0.5 * s, -0.5 * e2 * c], [0.5 * e1 * c, -0.5 * e1 * e2 * s]])
    if i == 1:
        return np.array([[0, 0], [complex(0, 1) * e1 * s, complex(0, 1) * e1 * e2 * c]])
    return np.array([[0, complex(0, -1) * e2 * s], [0, complex(0, 1) * e1 * e2 * c]])


def U3_hess(params, i, j):
    theta = params[0]
    phi = params[1]
    lam = params[2]
    c = math.cos(theta / 2)
    s = math.sin(theta / 2)
    e1 = cmath.exp(complex(0, phi))
    e2 = cmath.exp(complex(0, lam))
    if j < i:
        i, j = j, i
    if i == 0:
        if j == 0:
            return np.array([[c, -e2 * s], [e1 * s, e1 * e2 * c]])
        if j == 1:
            return np.array([[0, 0], [complex(0, 0.5) * e1 * c, complex(0, -0.5) * e1 * e2 * s]])
        return np.array([[0, complex(0, -0.5) * e2 * c], [0, complex(0, -0.5) * e1 * e2 * s]])
    if i == 1:
        if j == 1:
            return np.array([[0, 0], [- e1 * s, -e1 * e2 * c]])
        return np.array([[0, 0], [0, - e1 * e2 * c]])
    return np.array([[0, e2 * s], [0, -e1 * e2 * c]])


def phase(angle: float, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(angle)
    s = math.sin(angle)
    gate = np.array([[1, 0], [0, complex(c, s)]])
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def controlled_phase(angle: float, control: int, target: int, n_qubits: int) -> np.ndarray:
    if target == control:
        raise ValueError("control and target must be different")
    if control < target:
        gate = block_diag(np.eye(2 ** (n_qubits - control - 1)), phase(angle, target - control - 1, n_qubits - control - 1))
        for _ in range(control):
            gate = np.kron(np.eye(2), gate)
        return gate
    else:
        raise NotImplementedError("sorry control must be lower than target")


def Ry(angle: float, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    gate = np.array([[c, -s], [s, c]])
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def get_Ry_key(angle: float, n_qubits: int) -> Dict[str, np.ndarray]:
    return {"Ry" + str(i): Ry(angle, i, n_qubits) for i in range(4)}


def Rx(angle: float, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    gate = np.array([[c, complex(0, -s)], [complex(0, -s), c]])
    return multi_qubit_matrix(gate, i, n_qubits)


def get_Rx_key(angle: float, n_qubits: int) -> Dict[str, np.ndarray]:
    return {"Rx" + str(i): Ry(angle, i, n_qubits) for i in range(4)}


def multi_qubit_matrix(gate: np.ndarray, i: int, n_qubits: int) -> np.ndarray:
    return np.kron(np.kron(np.eye(2 ** i), gate), np.eye(2 ** (n_qubits - i - 1)))
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


H = np.array([[1, 1], [1, -1]]) * 0.5 ** 0.5
S = np.array([[1, 0], [0, complex(0, 1)]])
T = np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]])
V = Rx(math.pi / 2, 0, 1)

single_qubit_Clifford_T = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
       "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
       "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
       "X": np.array([[0, 1], [1, 0]])}

two_qubit_clifford_T = {s + "0": np.kron(single_qubit_Clifford_T[s], np.eye(2)) for s in single_qubit_Clifford_T.keys()}
two_qubit_clifford_T.update({s + "1": np.kron(np.eye(2), single_qubit_Clifford_T[s]) for s in single_qubit_Clifford_T})
two_qubit_clifford_T.update({"C0": cnot12, "C1": cnot21})


def clifford_T_gate_set(n_qubits: int):
    key = {s + str(i): multi_qubit_matrix(single_qubit_Clifford_T[s], i, n_qubits) for i in range(n_qubits) for s in "STHX"}
    key.update(get_cnot_key(n_qubits))
    return key


# def random_unitary():
#     phi1 = random.random() * math.pi * 2
#     phi2 = random.random() * math.pi * 2
#     theta = random.random() * math.pi * 2
#     print(phi1 / math.pi, phi2 / math.pi, theta / math.pi)
#     c = math.cos(theta)
#     s = math.sin(theta)
#     e1 = cmath.exp(complex(0, phi1))
#     e2 = cmath.exp(complex(0, phi2))
#     return np.array([[e1 * c, e2 * s], [- s / e2, c / e1]])


def random_unitary(n_qubits):
    return unitary_group.rvs(2 ** n_qubits)


def random_two_qubit_circuit():
    key = {"A1": np.kron(random_unitary(), np.eye(2)),
           "A2": np.kron(np.eye(2), random_unitary()),
           "A3": np.kron(random_unitary(), np.eye(2)),
           "A4": np.kron(np.eye(2), random_unitary()),
           "C0": cnot12,
           "C1": cnot21,
           "Rzt1": Rz(random.random() * 2 * math.pi, 0, 2),
           "Ryt2": Ry(random.random() * 2 * math.pi, 1, 2),
           "Ryt3": Ry(random.random() * 2 * math.pi, 1, 2)}
    return ["A1", "A2", "C1", "Rzt1", "Ryt2", "C0", "Ryt3", "C1", "A3", "A4"], key


def discrete_angle_key(angle_precision: int, n_qubits: int):
    x_key = {"Rx" + str(i) + ":" + str(k): Rx(k * math.pi * 2 ** (1 - angle_precision), i, n_qubits)
             for i in range(n_qubits) for k in range(1, 2 ** angle_precision)}
    y_key = {"Ry" + str(i) + ":" + str(k): Ry(k * math.pi * 2 ** (1 - angle_precision), i, n_qubits)
             for i in range(n_qubits) for k in range(1, 2 ** angle_precision)}
    z_key = {"Rz" + str(i) + ":" + str(k): Rz(k * math.pi * 2 ** (1 - angle_precision), i, n_qubits)
             for i in range(n_qubits) for k in range(1, 2 ** angle_precision)}
    return {**x_key, **y_key, **z_key}

