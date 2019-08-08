from openfermion.utils._unitary_cc import uccsd_singlet_generator
from openfermion.transforms import jordan_wigner
from openfermion import QubitOperator
#from tket_pauli_gadgets.noise_models import channels
from pytket import OpType, Circuit
from common_gates import multi_qubit_matrix, cnot, X, Y, Z, H, S, V, Rx, Ry, Rz, U1, U3
import math
import numpy as np
from functools import reduce
from tket_pauli_gadgets.chem import get_circuit
import scipy.optimize
import matplotlib.pyplot as plt


#single_noise, cnot_noise = channels(amplification=1000)

np.set_printoptions(edgeitems=10, linewidth=1000)


matrices_no_params = {OpType.Z: lambda i, n: multi_qubit_matrix(Z, i[0], n),
                      OpType.X: lambda i, n: multi_qubit_matrix(X, i[0], n),
                      OpType.Y: lambda i, n: multi_qubit_matrix(Y, i[0], n),
                      OpType.H: lambda i, n: multi_qubit_matrix(H, i[0], n),
                      OpType.S: lambda i, n: multi_qubit_matrix(S, i[0], n),
                      OpType.V: lambda i, n: multi_qubit_matrix(V, i[0], n),
                      OpType.CX: lambda i, n: cnot(i[0], i[1], n)}

matrices_with_params = {OpType.Rx: lambda i, n, params: Rx(params[0], i[0], n),
                        OpType.Ry: lambda i, n, params: Ry(params[0], i[0], n),
                        OpType.Rz: lambda i, n, params: Rz(params[0], i[0], n),
                        OpType.U1: lambda i, n, params: U1(params[0], i[0], n),
                        OpType.U3: lambda i, n, params: U3(params[0], params[1], params[2], i[0], n)}


n_qubits = 4


def matrix_list_product(matrices, default_size=None):
    if len(matrices) == 0:
        if default_size is None:
            default_size = 2 ** n_qubits
        return np.eye(default_size)
    return reduce(lambda x, y: x @ y, matrices, np.eye(matrices[0].shape[0]))

def instruction_to_unitary(instruction):
    t = instruction.op.get_type()
    if t in matrices_no_params:
        return matrices_no_params[t](instruction.qubits, n_qubits)
    elif t in matrices_with_params:
        return matrices_with_params[t](instruction.qubits, n_qubits,
                                       [p * math.pi for p in instruction.op.get_params()])
    else:
        raise ValueError("Unexpected instruction", instruction)


def circuit_to_unitary(circuit):
    return matrix_list_product([instruction_to_unitary(i) for i in circuit.get_commands()[::-1]])


def get_expectation(pauli, density):
    m = np.array([[1]])
    d = dict(pauli)
    for i in range(n_qubits):
        if i in d:
            if d[i] == "X":
                n = X
            elif d[i] == "Y":
                n = Y
            elif d[i] == "Z":
                n = Z
            else:
                raise ValueError("Unexpected Pauli label", d[i])
        else:
            n = np.eye(2)
        m = np.kron(m, n)
    expectation = np.einsum('ij,ji->', density, m)
    return expectation


def get_energy(packed_amplitudes):
    fermion_generator = uccsd_singlet_generator(packed_amplitudes, 4, 2)

    qubit_generator = jordan_wigner(fermion_generator)
    qubit_generator.compress()

    circuit = get_circuit(packed_amplitudes, 13)
    # circuit = Circuit(4)
    # circuit.X(0)
    # circuit.X(1)

    terms = {(): - 0.10973055650861605,
             ((0, 'Z'),): 0.1698845202255903,
             ((1, 'Z'),): 0.1698845202255903,
             ((2, 'Z'),): - 0.21886306765204747,
             ((3, 'Z'),): - 0.21886306765204747,
             ((0, 'Z'), (1, 'Z')): 0.1682119867202592,
             ((0, 'Z'), (2, 'Z')): 0.12005143070514901,
             ((0, 'Z'), (3, 'Z')): 0.1654943148544621,
             ((1, 'Z'), (2, 'Z')): 0.1654943148544621,
             ((1, 'Z'), (3, 'Z')): 0.12005143070514901,
             ((2, 'Z'), (3, 'Z')): 0.17395378774871575,
             ((0, 'X'), (1, 'X'), (2, 'Y'), (3, 'Y')): - 0.045442884149313106,
             ((0, 'X'), (1, 'Y'), (2, 'Y'), (3, 'X')): 0.045442884149313106,
             ((0, 'Y'), (1, 'X'), (2, 'X'), (3, 'Y')): 0.045442884149313106,
             ((0, 'Y'), (1, 'Y'), (2, 'X'), (3, 'X')): - 0.045442884149313106}

    u = circuit_to_unitary(circuit)
    start_density = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    start_density[0][0] = 1
    end_density = u @ start_density @ u.transpose().conjugate()

    return sum([terms[pauli] * get_expectation(pauli, end_density) for pauli in terms]).real

plt.figure()
ps = [x / 100 - 1 for x in range(400)]
plt.plot([p / math.pi for p in ps], [get_energy((0, p)) for p in ps])
plt.savefig("../graphs/energy.png")
plt.close()
params = [1, 0.5]
res = scipy.optimize.minimize(get_energy, params)
if res.success:
    print("optimal params", res.x)
    print("energy", get_energy(res.x))
else:
    print(res)
