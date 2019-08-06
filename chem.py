from pytket import PI
from pytket import Circuit

from typing import List, Tuple

from openfermion import uccsd_singlet_generator, jordan_wigner, uccsd_singlet_paramsize
from numpy import  ndarray, full

from pytket.qiskit import tk_to_dagcircuit
from qiskit.converters.dag_to_circuit import dag_to_circuit


def pauli_evolution(pauli: List[Tuple[int, str]], coeff: complex, circ: Circuit):
   """Appends the evolution circuit corresponding to a given Pauli tensor

   Args:
       pauli:
       coeff (complex):
       circ (Circuit):
   """
   # set up the correct basis
   all_qbs = list(zip(*pauli))[0]
   for qb_idx, p in pauli:
       if p == 'X':
           circ.H(qb_idx)
       elif p == 'Y':
           # angles in half-turns
           circ.Rx(qb_idx, 0.5)

   # cnot cascade
   cx_qb_pairs = list(zip(sorted(all_qbs)[:-1], sorted(all_qbs)[1:]))
   for pair in cx_qb_pairs:
       circ.CX(pair[0], pair[1])

   # rotation (convert angle from radians to half-turns)
   circ.Rz(all_qbs[-1], (2 * coeff.imag) / PI)

   # reverse cascade and revert basis
   cx_qb_pairs = list(zip(sorted(all_qbs)[:-1], sorted(all_qbs)[1:]))
   for pair in reversed(cx_qb_pairs):
       circ.CX(pair[0], pair[1])

   all_qbs = list(zip(*pauli))[0]

   for qb_idx, p in pauli:
       if p == 'X':
           circ.H(qb_idx)
       elif p == 'Y':
           circ.Rx(qb_idx, -0.5)


def generate_hf_wavefunction_circuit(n_qubit: int, n_alpha_electron: int, n_beta_electron: int):
   """
   Args:
       n_qubit (int):
       n_alpha_electron (int):
       n_beta_electron (int):
   """
   circuit = Circuit(n_qubit)

   for i in range(n_alpha_electron):
       idx_alpha = 2 * i
       circuit.X(idx_alpha)

   for i in range(n_beta_electron):
       idx_beta = 2 * i + 1
       circuit.X(idx_beta)

   return circuit


# entangle
def get_mb_wavefunction_circuit(packed_amplitudes: ndarray, n_spin_orb, n_electron):

   fermion_generator = uccsd_singlet_generator(packed_amplitudes, n_spin_orb,
                                               n_electron)

   qubit_generator = jordan_wigner(fermion_generator)
   qubit_generator.compress()

   mb_wavefunction_circuit = generate_hf_wavefunction_circuit(n_spin_orb, n_electron//2, n_electron//2)

   for pauli, coeff in qubit_generator.terms.items():
       pauli_evolution(pauli, coeff, mb_wavefunction_circuit)

   return mb_wavefunction_circuit


n_qubit = 4
n_electron = 2
n_amplitudes = uccsd_singlet_paramsize(n_qubit, n_electron)

amplitudes = full(n_amplitudes, 0.5)

circ = get_mb_wavefunction_circuit(amplitudes, n_qubit, n_electron)


def instructions_to_circuit(instructions):
    c = Circuit(n_qubit)
    for inst in instructions:
        t = inst.op.get_type()
        c.add_operation(t, inst.op.get_params(), inst.qubits)
    return c


#circ = instructions_to_circuit(circ.get_commands()[:24])

#print(circ.n_gates)

#print(dag_to_circuit(tk_to_dagcircuit(circ)))