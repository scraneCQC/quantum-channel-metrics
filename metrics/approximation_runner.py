from pytket import Circuit
from typing import Iterable, Any, List, Dict, Callable
import math


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


final_circuits = {a: Circuit(1) for a in "IXYZ"}
final_circuits["X"].H(0)
final_circuits["Y"].Sdg(0)
final_circuits["Y"].H(0)


def get_pauli_expectation(c: Circuit, initial_circuit: Circuit, pauli_string: str, backend, *,
                          shots: int = 100) -> float:
    n_qubits = len(pauli_string)
    circuit = initial_circuit.copy()
    circuit.add_circuit(c.copy(), list(range(n_qubits)))
    black_box_exp = backend.get_pauli_expectation_value(circuit,
                [(i, pauli_string[i]) for i in range(len(pauli_string)) if pauli_string[i] != "I"], shots=shots).real
    return black_box_exp

