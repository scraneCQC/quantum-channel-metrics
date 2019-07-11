from pytket.backends.ibm import AerStateBackend, AerBackend, IBMQBackend
from pytket import Circuit
from qiskit.quantum_info.operators import Operator
from pytket import Transform
from qiskit.providers.aer.noise.errors import amplitude_damping_error, phase_damping_error, depolarizing_error
from qiskit.providers.aer.noise import NoiseModel
from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit
import time


gridsynth = open("pi_over_128.txt")
lines = gridsynth.readlines()
gridsynth.close()
op_strings = list(map(lambda x: x.split(" ")[1].strip(), lines))


def circuit_from_string(op_string):
    ops = {"S": lambda c: c.S(0),
           "T": lambda c: c.T(0),
           "H": lambda c: c.H(0),
           "X": lambda c: c.X(0),
           "W": lambda c: None,     # TODO: how to implement this in tket?
                                    # It's a scalar multiplication by e^i*pi/4, so doesn't affect the density matrix
           "I": lambda c: None}

    circuit = Circuit(1)
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

    noisy_backend = AerBackend(my_noise_model)
    return noisy_backend


def run_circuit(op_string, p1, gamma1, gamma2, shots=100):
    circuit = circuit_from_string(op_string)
    circuit.measure_all()

    # Transform.RebaseToQiskit().apply(circuit)
    # print(dag_to_circuit(tk_to_dagcircuit(circuit)))

    noisy_backend = make_noisy_backend(p1, gamma1, gamma2)

    noisy_shots = noisy_backend.get_counts(circuit=circuit, shots=shots)
    if (0,) not in noisy_shots:
        noisy_shots[(0,)] = 0
    if (1,) not in noisy_shots:
        noisy_shots[(1,)] = 0
    return noisy_shots


for s in op_strings:
    print(run_circuit(s, 0.01, 0.01, 0.01))
