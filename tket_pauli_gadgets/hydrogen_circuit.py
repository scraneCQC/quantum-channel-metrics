from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from tket_pauli_gadgets.noise_models import channels
from tket_pauli_gadgets.chem import get_circuit
import numpy as np
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
from pytket import Circuit
import matplotlib.pyplot as plt


np.set_printoptions(edgeitems=10, linewidth=1000)

single_noise, cnot_noise = channels(amplification=10000)

params = [-4.876143648314624e-05, 0.057384102234558684]
params = [0.4, 20]
complete_circuit = get_circuit(params, 13)
# print(dag_to_circuit(tk_to_dagcircuit(complete_circuit)))

rewriter = RewriteTket(get_circuit(params, 13), single_noise, cnot_noise)
target = rewriter.target

fids = []
fid_opt = []

for i in range(13):
    # if i == 3:
    #     fids.append(fids[-1])
    #     fid_complete.append(fid_complete[-1])
    #     continue
    short_circ = get_circuit(params, i)
    rewriter.set_circuit(short_circ)
    f = rewriter.reduce()
    fids.append(f[0])
    fid_opt.append(f[1])git a

plt.figure()
plt.plot(fids)
plt.plot(fid_opt)
plt.xlabel("Number of Pauli gadgets")
plt.ylabel("Fidelity")
plt.tight_layout()
plt.savefig("../graphs/number_of_gadgets_hydrogen.png")
plt.close()



