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
from itertools import product
from typing import Iterable, Any, List, Dict, Callable


def load_file(filename: str) -> List[str]:
    gridsynth = open(filename)
    lines = gridsynth.readlines()
    gridsynth.close()
    op_strings = list(map(lambda x: x.strip(), lines))
    return op_strings


ops = {"S": lambda c: c.S(0),
       "T": lambda c: c.T(0),
       "H": lambda c: c.H(0),
       "X": lambda c: c.X(0),
       "W": lambda c: None,     # Doesn't affect any possible measurements, just a global phase
       "I": lambda c: None}


def circuit_from_string(op_string: Iterable[Any], *, n_qubits: int = 1, key: Dict[Any, Callable] = None) -> Circuit:
    if key is None:
        key = ops
    circuit = Circuit(n_qubits)
    for s in op_string[::-1]:
        key[s](circuit)
    return circuit


def make_noisy_backend(p1: float, gamma1: float, gamma2: float) -> AerBackend:
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


def run_circuit(op_string: Iterable[Any], p1: float, gamma1: float, gamma2: float, shots: int = 1000, *,
                prep_circuit: Circuit = None, key: Dict[Any, Callable] = None, n_qubits: int = 1) -> Dict[Any, int]:
    circuit = circuit_from_string(op_string, prep_circuit=prep_circuit, key=key)
    circuit.measure_all()

    # Transform.RebaseToQiskit().apply(circuit)
    # print(dag_to_circuit(tk_to_dagcircuit(circuit)))

    noisy_backend = make_noisy_backend(p1, gamma1, gamma2)

    noisy_shots = noisy_backend.get_counts(circuit=circuit, shots=shots)
    all_outcomes = tuple(product((0, 1), repeat=n_qubits))
    for o in all_outcomes:
        if o not in noisy_shots:
            noisy_shots[o] = 0
    return noisy_shots


def get_pauli_expectation(circuit_string: Iterable[Any], initial_circuit: Circuit, pauli_string: str, n_qubits: int, *,
                          p1: float = 0, gamma1: float = 0, gamma2: float = 0, shots: int = 100,
                          key: Dict[Any, Callable] = None) -> float:
    if pauli_string == "I" * n_qubits:
        return 1
    circuit = initial_circuit.copy()
    circuit.add_circuit(circuit_from_string(circuit_string, n_qubits=n_qubits, key=key), list(range(n_qubits)))
    noisy_backend = make_noisy_backend(p1, gamma1, gamma2)
    return noisy_backend.get_pauli_expectation_value(circuit,
                [(i, pauli_string[i]) for i in range(len(pauli_string)) if pauli_string[i] != "I"], shots=shots).real


final_circuits = {a: Circuit(1) for a in "IXYZ"}
final_circuits["X"].H(0)
final_circuits["Y"].Sdg(0)
final_circuits["Y"].H(0)


def get_pauli_expectation_v2(circuit_string: Iterable[Any], initial_circuit: Circuit, pauli_string: str, *,
                             p1: float = 0, gamma1: float = 0, gamma2: float = 0, shots: int = 100,
                             key: Dict[Any, Callable] = None) -> float:
    n_qubits = len(pauli_string)
    if pauli_string == "I" * n_qubits:
        return 1
    circuit = initial_circuit.copy()
    circuit.add_circuit(circuit_from_string(circuit_string, n_qubits=n_qubits, key=key), list(range(n_qubits)))
    for i in range(n_qubits):
        circuit.add_circuit(final_circuits[pauli_string[i]].copy(), [i])
    circuit.measure_all()
    noisy_backend = make_noisy_backend(p1, gamma1, gamma2)
    start = time.time()
    noisy_shots = noisy_backend.get_counts(circuit, shots)
    end = time.time()
    print(end-start)
    return sum(v * (-1) ** sum(k) for k, v in noisy_shots.items())/shots
