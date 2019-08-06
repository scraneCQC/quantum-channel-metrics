from pytket.qiskit import dagcircuit_to_tk
from qiskit.converters.circuit_to_dag import circuit_to_dag
from qiskit import QuantumCircuit


fn = "preopt.qasm"
qc = QuantumCircuit.from_qasm_file(fn)
dag = circuit_to_dag(qc)
tkcirc_pre_opt = dagcircuit_to_tk(dag)

