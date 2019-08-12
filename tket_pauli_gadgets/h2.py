from openfermion.utils._unitary_cc import uccsd_singlet_generator
from openfermion.transforms import jordan_wigner
from openfermion import QubitOperator
#from tket_pauli_gadgets.noise_models import channels
from common_gates import X, Y, Z
import numpy as np
from tket_pauli_gadgets.chem import get_circuit
import scipy.optimize
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tket_pauli_gadgets.converter import converter
import math


#single_noise, cnot_noise = channels(amplification=1000)

np.set_printoptions(edgeitems=10, linewidth=1000)


n_qubits = 4
converter.n_qubits = n_qubits


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

    u = converter.circuit_to_unitary(circuit)
    start_density = np.zeros((2 ** n_qubits, 2 ** n_qubits))
    start_density[0][0] = 1
    end_density = u @ start_density @ u.transpose().conjugate()

    return sum([terms[pauli] * get_expectation(pauli, end_density) for pauli in terms]).real


print(get_energy((math.pi, 0.05)))

quit()

plt.figure()
ps = [x / 20 - 2.5 for x in range(100)]
plt.imsave("../graphs/energy_grid.png", [[get_energy((p1, p2)) for p2 in ps] for p1 in ps], cmap=cm.gist_rainbow)
# plt.plot([p1 / math.pi for p1 in ps], [get_energy((0, p1)) for p1 in ps])
# plt.savefig("../graphs/energy_grid_0.png")
plt.close()
params = [1, 0.5]
res = scipy.optimize.minimize(get_energy, params)
if res.success:
    print("optimal params", res.x)
    print("energy", get_energy(res.x))
else:
    print(res)
