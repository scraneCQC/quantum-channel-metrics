from pytket.qiskit import dagcircuit_to_tk
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit import QuantumCircuit
from metrics.J_fidelity import f_pro_simulated
import numpy as np
from functools import reduce
from tket_pauli_gadgets.tket_circuit_rewriter import matrices_with_params, matrices_no_params
import math
from pytket.backends.ibm import AerBackend
from tket_pauli_gadgets.noise_models import amplified_qiskit_model
from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket, cleanup
from tket_pauli_gadgets.noise_models import channels
from chem import circ



fn = "preopt.qasm"
qc = QuantumCircuit.from_qasm_file(fn)
dag = circuit_to_dag(qc)
tkcirc_pre_opt = dagcircuit_to_tk(dag)

fn = "simon_h2.qasm"
qc = QuantumCircuit.from_qasm_file(fn)
dag = circuit_to_dag(qc)
tkcirc_post_opt = dagcircuit_to_tk(dag)

n_qubits = 4


# based on ibmqx4 averages
single_noise, cnot_noise = channels(amplification=100)

rewriter = RewriteTket(circ, single_noise, cnot_noise, verbose=True)

print(rewriter.fidelity(rewriter.instructions))


#
# def matrix_list_product(matrices, default_size=1):
#     if len(matrices) == 0:
#         return np.eye(default_size)
#     return reduce(lambda x, y: x @ y, matrices, np.eye(matrices[0].shape[0]))
#
#
# def instruction_to_unitary(instruction):
#     t = instruction.op.get_type()
#     if t in matrices_no_params:
#         return matrices_no_params[t](instruction.qubits, n_qubits)
#     elif t in matrices_with_params:
#         return matrices_with_params[t](instruction.qubits, n_qubits, [p * math.pi for p in instruction.op.get_params()])
#     else:
#         raise ValueError("Unexpected instruction", instruction)
#
#
# unitary = matrix_list_product(
#             [instruction_to_unitary(inst) for inst in tkcirc_pre_opt.get_commands()[::-1]])
#
# noise_model = amplified_qiskit_model('ibmqx4')
# backend = AerBackend(noise_model)
#
# f = f_pro_simulated(tkcirc_pre_opt, unitary, backend)
# print(f)
#
