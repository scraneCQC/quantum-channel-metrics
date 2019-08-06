from pytket import Circuit
import random


def U(s, n_qubits):
    c = Circuit(n_qubits)
    for i in range(len(s)):
        if s[i] == "X":
            c.H(i)
        elif s[i] == "Y":
            c.Rx(i, 0.5)
        elif s[i] == "Z":
            pass
    return c


def phi(qubits, alpha, n_qubits):
    c = Circuit(n_qubits)
    for i in range(len(qubits) - 1):
        c.CX(qubits[i], qubits[i + 1])
    c.Rz(qubits[-1], alpha)
    for i in range(len(qubits) - 1):
        c.CX(qubits[-i - 2], qubits[-i - 1])
    return c


def pauli_gadget(alpha, s, n_qubits):
    c1 = U(s, n_qubits)
    c2 = phi([x for x in range(n_qubits) if s[x] != "I"], alpha, n_qubits)
    c = Circuit(n_qubits)
    c.add_circuit(c1, list(range(n_qubits)))
    c.add_circuit(c2, list(range(n_qubits)))
    c.add_circuit(c1.dagger(), list(range(n_qubits)))
    return c


def random_pauli_gadget(n_qubits):
    s = "".join(random.choices("XYZ", k=n_qubits))
    alpha = random.random() * 2
    print("s", s)
    print("alpha", alpha)
    return pauli_gadget(alpha, s, n_qubits)
