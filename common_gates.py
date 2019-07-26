import numpy as np
import math
import cmath
from typing import Dict
from scipy.linalg import block_diag


cnot12 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
cnot21 = np.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])


def cnot(i: int, j: int, n_qubits: int) -> np.ndarray:
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


cnot_key1 = {"cnot" + str(i) + str(i + 1): cnot(i, i + 1, 4) for i in range(3)}
cnot_key2 = {"cnot" + str(i + 1) + str(i): cnot(i + 1, i, 4) for i in range(3)}
cnot_key = {**cnot_key1, **cnot_key2, "cnot01-23": np.kron(cnot12, cnot12), "cnot01-32": np.kron(cnot12, cnot21)}


def get_cnot_key(n_qubits):
    key1 = {"cnot" + str(i) + str(i + 1): cnot(i, i + 1, n_qubits) for i in range(n_qubits - 1)}
    key2 = {"cnot" + str(i + 1) + str(i): cnot(i + 1, i, n_qubits) for i in range(n_qubits - 1)}
    return {**key1, **key2}


def Rz(angle: float, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    gate = np.array([[complex(c, -s), 0], [0, complex(c, s)]])
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
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def get_Rx_key(angle: float, n_qubits: int) -> Dict[str, np.ndarray]:
    return {"Rx" + str(i): Ry(angle, i, n_qubits) for i in range(4)}


def multi_qubit_matrix(gate: np.ndarray, i: int, n_qubits: int) -> np.ndarray:
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


H = np.array([[1, 1], [1, -1]]) * 0.5 ** 0.5

single_qubit_Clifford_T = {"S": np.array([[1, 0], [0, complex(0, 1)]]),
       "T": np.array([[1, 0], [0, complex(0.5 ** 0.5, 0.5 ** 0.5)]]),
       "H": 0.5 ** 0.5 * np.array([[1, 1], [1, -1]]),
       "X": np.array([[0, 1], [1, 0]])}

two_qubit_clifford_T = {s + "0": np.kron(single_qubit_Clifford_T[s], np.eye(2)) for s in single_qubit_Clifford_T.keys()}
two_qubit_clifford_T.update({s + "1": np.kron(np.eye(2), single_qubit_Clifford_T[s]) for s in single_qubit_Clifford_T})
two_qubit_clifford_T.update({"C0": cnot12, "C1": cnot21})

