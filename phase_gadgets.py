import numpy as np
import math
from typing import Dict


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


def Rz(angle: float, i: int, n_qubits: int) -> np.ndarray:
    c = math.cos(angle / 2)
    s = math.sin(angle / 2)
    gate = np.array([[complex(c, -s), 0], [0, complex(c, s)]])
    for _ in range(i):
        gate = np.kron(np.eye(2), gate)
    for _ in range(n_qubits - i - 1):
        gate = np.kron(gate, np.eye(2))
    return gate


def get_Rz_key(angle: float) -> Dict[str, np.ndarray]:
    return {"Rz" + str(i): Rz(angle, i, 4) for i in range(4)}


def get_key(angle: float) -> Dict[str, np.ndarray]:
    return {**get_Rz_key(angle), **cnot_key}