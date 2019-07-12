from pytket.backends.ibm import AerStateBackend, AerBackend, IBMQBackend
from pytket import Circuit
from qiskit.quantum_info.operators import Operator
from pytket import Transform
from qiskit.providers.aer.noise.errors import amplitude_damping_error, phase_damping_error, depolarizing_error
from qiskit.providers.aer.noise import NoiseModel
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
import time
import numpy as np
from openfermion.ops import QubitOperator
from Pauli import one_qubit_diracs


def load_file(filename):
    gridsynth = open(filename)
    lines = gridsynth.readlines()
    gridsynth.close()
    op_strings = list(map(lambda x: x.strip(), lines))
    return op_strings


def circuit_from_string(op_string, *, prep_circuit=None):
    ops = {"S": lambda c: c.S(0),
           "T": lambda c: c.T(0),
           "H": lambda c: c.H(0),
           "X": lambda c: c.X(0),
           "W": lambda c: None,     # Doesn't affect any possible measurements, just a global phase
           "I": lambda c: None}

    circuit = Circuit(1)
    if prep_circuit is not None:
        prep_circuit(circuit)
    for s in op_string[::-1]:
        ops[s](circuit)

    return circuit


def make_noisy_backend(p1, gamma1, gamma2):
    my_noise_model = NoiseModel()

    amp_error = amplitude_damping_error(gamma1)
    phase_error = phase_damping_error(gamma2)
    dep_error = depolarizing_error(p1, 1)

    # Not sure what order these should go in or if it matters
    total_error = amp_error.compose(phase_error.compose(dep_error))
    my_noise_model.add_all_qubit_quantum_error(total_error, ['u1', 'u2', 'u3'])
    # S and T are u1, H is u2, X is u3

    noisy_backend = AerBackend(my_noise_model)
    return noisy_backend


def run_circuit(op_string, p1, gamma1, gamma2, shots=1000, observable=None, *, prep_circuit=None):
    circuit = circuit_from_string(op_string, prep_circuit=prep_circuit)
    if observable is None:
        circuit.measure_all()

    # Transform.RebaseToQiskit().apply(circuit)
    # print(dag_to_circuit(tk_to_dagcircuit(circuit)))

    noisy_backend = make_noisy_backend(p1, gamma1, gamma2)

    if observable is not None:
        return noisy_backend.get_operator_expectation_value(circuit, observable, shots=shots)

    noisy_shots = noisy_backend.get_counts(circuit=circuit, shots=shots)
    if (0,) not in noisy_shots:
        noisy_shots[(0,)] = 0
    if (1,) not in noisy_shots:
        noisy_shots[(1,)] = 0
    return noisy_shots


def decompose_operator(observable_matrix):
    coefficients = [complex(np.trace(observable_matrix @ d)) / 2 for d in one_qubit_diracs[1:]]
    q = QubitOperator()
    for i in range(3):
        if coefficients[i] != 0:
            q += QubitOperator("XYZ"[i]+"0", coefficients[i])
    return q


def get_expectation(observable_matrix, circuit_string, *, prep_circuit=None, p1=0, gamma1=0, gamma2=0, shots=1000000):
    pauli_bits = run_circuit(circuit_string, p1, gamma1, gamma2, observable=decompose_operator(observable_matrix), prep_circuit=prep_circuit, shots=shots)
    identity_bit = np.trace(observable_matrix) / 2
    return (pauli_bits + identity_bit).real

