from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket, cleanup
from tket_pauli_gadgets.noise_models import channels
from tket_pauli_gadgets.chem import get_circuit
import numpy as np
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
import matplotlib.pyplot as plt


np.set_printoptions(edgeitems=10, linewidth=1000)

amplification = 10
single_noise, cnot_noise = channels(amplification=amplification)
params = [0, 0.05]

circ = get_circuit(params, 13)
rewriter = RewriteTket(circ, single_noise, cnot_noise, verbose=False)

fid_none = []
fid_tket = []
fid_opt = []

for i in range(13):
    print(i)
    short_circ = get_circuit(params, i)
    rewriter.set_circuit(short_circ.copy())
    rewriter.original_fidelity = rewriter.fidelity(rewriter.instructions)
    fid_none.append(rewriter.original_fidelity)
    rewriter.set_circuit(short_circ)
    f = rewriter.reduce()
    fid_tket.append(f[0])
    fid_opt.append(f[1])


plt.figure()
plt.plot(fid_none, label="no optimization")
plt.plot(fid_tket, label="tket")
plt.plot(fid_opt, label="tket + small angle")
plt.xlabel("Number of Pauli gadgets")
plt.ylabel("Fidelity")
plt.legend()
plt.tight_layout()
plt.savefig("../graphs/number_of_gadgets_hydrogen_noise_amp_" + str(amplification) + "_params_" + str(params) + ".png")
plt.close()


