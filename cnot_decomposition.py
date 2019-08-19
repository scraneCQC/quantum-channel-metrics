import math
import numpy as np
from itertools import product
from pytket import Circuit
import common_gates as gates
from tket_pauli_gadgets.converter import converter
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit


def suggest_cnot_unitary(unitary):
    dim = unitary.shape[0]
    basis_rows = [r for r in np.eye(dim)]
    new_rows = [basis_rows[np.argmax(np.abs(u_row))] for u_row in unitary]
    new = np.vstack(new_rows)
    if (new @ new.transpose() != np.eye(dim)).any():
        raise ValueError("I don't know to make that unitary")
    return new


def bools(n_qubits):
    return list(product((0, 1), repeat=n_qubits))


def matrix_to_function(m, n_qubits):
    inputs = bools(n_qubits)
    d = dict()
    for i in range(2 ** n_qubits):
        d[inputs[i]] = inputs[np.argmax(m.transpose()[i])]

    def f(b):
        return d[b]

    return f


def mnot(cs, ds, t):
    def f(b):
        if all([b[i] for i in cs]) and all(not b[i] for i in ds):
            c = list(b)
            c[t] = 1 - c[t]
            return tuple(c)
        return b
    return f


def cnot(i, j):
    def f(b):
        if b[i]:
            c = list(b)
            c[j] = 1 - c[j]
            return tuple(c)
        return b
    return f


def anot(i):
    def f(b):
        c = list(b)
        c[i] = 1 - c[i]
        return tuple(c)
    return f


def compose(f, g):
    def h(b):
        return f(g(b))
    return h


def identity(b):
    return b


def function_to_mnots(f, n_qubits):
    desc = []
    g = f
    z = tuple(0 for _ in range(n_qubits))
    for i in range(n_qubits):
        if f(z)[i]:
            desc.append("N:"+str(i))
            g = compose(anot(i), g)
    for b in bools(n_qubits):
        if b == z:
            continue
        first = b.index(1)
        last = n_qubits - b[::-1].index(1) - 1
        cs = [i for i in range(n_qubits) if first <= i <= last and b[i]]
        ds = [i for i in range(n_qubits) if first <= i <= last and not b[i]]
        for i in range(n_qubits):
            if g(b)[i] != b[i]:
                desc.append((cs + ds, i))
                g = compose(mnot(cs, ds, i), g)
    return desc


def mnot_templates(n_controls):
    if n_controls == 0:
        return Circuit(1)
    c1 = Circuit(2)
    c1.CX(0, 1)
    c1.add_circuit(c1.copy(), [0, 1])
    if n_controls == 1:
        return c1
    c2 = Circuit(3)
    c2.CX(1, 2)
    c2.CX(0, 1)
    c2.CX(1, 2)
    c2.CX(0, 1)
    c2.CX(0, 2)
    circs = [c1, c2]
    for i in range(3, n_controls + 1):
        c = Circuit(i + 1)
        c.CX(i - 1, i)
        c.CX(i - 1, i)
        c.add_circuit(circs[-1], list(range(i)))
        c.CX(i - 1, i)
        c.CX(i - 1, i)
        c.add_circuit(circs[-1], list(range(i)))
        c.add_circuit(circs[-1], list(range(i - 1)) + [i])
        c.add_circuit(circs[-1], list(range(i - 1)) + [i])
        circs.append(c)
    return circs


def greedy(f, n_qubits):
    all_funs = [anot(i) for i in range(n_qubits)] + \
               [cnot(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    descriptions = ["N:"+str(i) for i in range(n_qubits)] + \
                   ["C:" + str(i) + ":" + str(j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    g = f
    desc = []
    while any(g(b) != b for b in bools(n_qubits)):
        print(desc)
        possible = [(d, agreement(compose(fun, g), identity, n_qubits), fun) for d, fun in zip(descriptions, all_funs)]
        best = max(possible, key=lambda x: x[1])
        desc = desc + [best[0]]
        g = compose(best[2], g)
    return desc


def agreement(f, g, n_qubits):
    return sum(sum(x == y for x, y in zip(f(b), g(b))) for b in bools(n_qubits))


def decompose_multi_controlled_cnot(controls, target):
    pass


def decompose_unitary(unitary):
    n_qubits = int(math.log(unitary.shape[0], 2))
    all_funs = [gates.multi_qubit_matrix(gates.X, i, n_qubits) for i in range(n_qubits)] + \
               [gates.cnot(i, j, n_qubits) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    descriptions = ["N:" + str(i) for i in range(n_qubits)] + \
                   ["C:" + str(i) + ":" + str(j) for i in range(n_qubits) for j in range(n_qubits) if i != j]
    g = unitary
    desc = []
    best = (None, None, None)
    while len(desc) == 0 or best[0] != desc[-1]:
        print(desc)
        possible = [(d, np.einsum('ij,ji->', unitary, g), fun) for d, fun in zip(descriptions, all_funs)]
        best = max(possible, key=lambda x: x[1])
        desc = desc + [best[0]]
        g = best[2] @ g
    return desc


def cnot_unitary_to_circuit(u, n_qubits):
    # desc = greedy(matrix_to_function(u, n_qubits), n_qubits)  # Doesn't converge, don't run this
    desc = decompose_unitary(u)
    c = Circuit(n_qubits)
    for d in desc:
        if d[0] == "N":
            c.X(int(d[2:]))
        elif d[0] == "C":
            c.CX(int(d.split(":")[1]), int(d.split(":")[2]))
    return c


def approximate_with_cnots(unitary):
    n_qubits = int(math.log(unitary.shape[0], 2))
    mnots = function_to_mnots(matrix_to_function(suggest_cnot_unitary(unitary), n_qubits), n_qubits)
    circs = mnot_templates(n_qubits)
    c = Circuit(n_qubits)
    for mnot in mnots:
        n_controls = len(mnot[0])
        c.add_circuit(circs[n_controls - 1], mnot[0] + [mnot[1]])
    return c

