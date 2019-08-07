from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket, cleanup
from tket_pauli_gadgets.noise_models import channels
from tket_pauli_gadgets.chem import get_circuit
import numpy as np
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from pytket import Circuit, Transform, OpType
import matplotlib.pyplot as plt


np.set_printoptions(edgeitems=10, linewidth=1000)

amplification = 2001
single_noise, cnot_noise = channels(amplification=amplification)
params = [-4.876143648314624e-05, 0.057384102234558684]


circ = get_circuit(params, 13)
rewriter = RewriteTket(circ, single_noise, cnot_noise, verbose=False)

none = []
fid_basic_tket = []
fid_opt = []

for i in range(13):
    print(i)
    short_circ = get_circuit(params, i)
    none.append(rewriter.fidelity(short_circ.get_commands()))
    rewriter.set_circuit(short_circ)
    f = rewriter.reduce()
    print(f)
    fid_basic_tket.append(f[0])
    fid_opt.append(f[1])

print(none)
print(fid_basic_tket)
print(fid_opt)

plt.figure()
plt.plot(none, label="no optimization")
plt.plot(fid_basic_tket, label="basic tket")
plt.plot(fid_opt, label="basic tket + small angle")
plt.xlabel("Number of Pauli gadgets")
plt.ylabel("Fidelity")
plt.legend()
plt.tight_layout()
plt.savefig("../graphs/number_of_gadgets_hydrogen_noise_amp_" + str(amplification) + ".png")
plt.close()



