import sys

sys.path.append("/Users/nathanfitzpatrick/Documents/GitHub/Chemistry/eumen")

from pytket import Circuit
from pytket._circuit import optimise_pre_routing, optimise_post_routing
from utils import create_psi4_molecule, get_molecular_hamiltonian
from utils import hf_wavefunction, pauli_evolution
from pytket.backends import AerBackend, IBMQBackend
from pytket._transform import Transform
from pytket.qiskit import tk_to_dagcircuit

from qiskit.converters import dag_to_circuit

import psi4
from openfermion.hamiltonians import MolecularData
from openfermion.transforms import jordan_wigner
from openfermion.utils._unitary_cc import uccd_singlet_generator, uccsd_singlet_generator

from openfermion_uccsd_spin import *

import matplotlib
matplotlib.use('TkAgg')

bond_length = 0.75
geometry = [('H', (0., 0., 0.)), ('H', (0., 0., bond_length))]
basis = 'sto-3g'
multiplicity = 1
charge = 0
description = str(bond_length)

active_space = False
docc_orbitals = [0]
active_orbitals = [0,1]
n_docc_orbitals = 0
n_active_orbitals = 2

mol = MolecularData(geometry, basis, multiplicity, charge, description)
psi4_mol = create_psi4_molecule(mol, geometry, multiplicity, charge)

if mol.multiplicity == 1:
    psi4.set_options({'reference': 'rhf'})
    psi4.set_options({'guess': 'sad'})
else:
    psi4.set_options({'reference': 'rohf'})
    psi4.set_options({'guess': 'gwh'})

psi4.set_options({'basis': basis})
psi4.set_options({
    'freeze_core':'false',
    'fail_on_maxiter':'true',
    'df_scf_guess':'false',
    'opdm':'true', 'tpdm':'true',
    'soscf':'false',
    'scf_type':'pk',
    'maxiter':'1e6',
    'num_amps_print':'1e6',
    'r_convergence':'1e-6',
    'd_convergence':'1e-6',
    'e_convergence':'1e-6',
    'ints_tolerance':'EQUALITY_TOLERANCE',
    'damping_percentage':'0'
})

energy, wavefunction = psi4.energy('scf', return_wfn=True, molecule=psi4_mol)
hamiltonian = get_molecular_hamiltonian(mol, psi4_mol, energy, wavefunction,
                                        docc_orbitals, active_orbitals,
                                        active_space)
qubit_hamiltonian = jordan_wigner(hamiltonian)
qubit_hamiltonian.compress()

if active_space == False:
    n_qubits = mol.n_qubits
    n_electrons = mol.n_electrons
else:
    n_qubits = 2 * n_active_orbitals
    n_electrons = 2 * n_docc_orbitals

# backend = IBMQBackend("ibmqx4")
backend = AerBackend()

packed_amplitudes = [-4.876143648314624e-05, 0.057384102234558684]
fermion_generator = uccsd_singlet_generator(packed_amplitudes,
                       n_qubits,
                        n_electrons)

qubit_generator = jordan_wigner(fermion_generator)
qubit_generator.compress()

alpha_electrons = 1
beta_electrons = 1

#qubit_generator = uccd_general_evolution(packed_amplitudes, alpha_electrons, beta_electrons, n_qubits)

circ = Circuit(n_qubits)

hf_wavefunction(circ, n_electrons, multiplicity)

print (qubit_generator)

for pauli, coeff in qubit_generator.terms.items():
    pauli_evolution(pauli, coeff, circ)

print("Pre-Opt Qasm Circuit")        
dag = tk_to_dagcircuit(circ)
print(dag.qasm(qeflag=True))

print(circ.depth())
qc = dag_to_circuit(dag)
print(qc)

#optimise_pre_routing(circ)
Transform.get_transform_instance("PauliGadget_Opt").apply(circ)
optimise_post_routing(circ)

print("Post-Opt Qasm Circuit")        
dag = tk_to_dagcircuit(circ)
print(dag.qasm(qeflag=True))

print(circ.depth())
qc = dag_to_circuit(dag)
print(qc)

energy = 0
for pauli, coeff in qubit_hamiltonian.terms.items():

    shot_exp = backend.get_pauli_expectation_value(circ, pauli, 8000)
    shot_energy = coeff*shot_exp

    print(pauli, coeff, shot_exp, shot_energy )
    
    energy += coeff*shot_exp

print(energy.real)
