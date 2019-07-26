from common_gates import controlled_phase, H, multi_qubit_matrix
import math


def generate_circuit(n_qubits: int):
    key = {"H" + str(i): multi_qubit_matrix(H, i, n_qubits) for i in range(n_qubits)}
    for i in range(n_qubits -1, 0, -1):
        key.update({"phase-c:" + str(k) + "-t:" + str(i):
                        controlled_phase(math.pi / 2 ** k, k, i, n_qubits)
                    for k in range(i)})
    circuit = sum([["H"+str(i)] + ["phase-c:" + str(k) + "-t:" + str(i) for k in range(i - 1, -1, -1)]
                   for i in range(n_qubits - 1, -1, -1)], [])
    return circuit, key
