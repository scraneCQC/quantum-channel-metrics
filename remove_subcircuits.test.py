from remove_subcircuits import *
import random
from noise import depolarising_channel
from common_gates import discrete_angle_key, get_cnot_key
from metrics.J_fidelity import ProcessFidelityFinder

f_pro_experimental, f_pro

n_qubits = 3
precision = 4
key = discrete_angle_key(precision, n_qubits)
key.update(get_cnot_key(n_qubits))
weights = [1 for _ in range(3 * n_qubits * (2 ** precision - 1))] + [2 ** precision for _ in range(2 * (n_qubits - 1))]
noise = [depolarising_channel(0.01, n_qubits)]
fid = ProcessFidelityFinder(n_qubits)
f_pro_experimental, f_pro = fid.f_pro_experimental, fid.f_pro


def run():
    circuit1 = random.choices(list(key.keys()), k=10, weights=weights)

    remover = SubcircuitRemover(circuit1, key, noise, n_qubits=n_qubits, verbose=True)
    u1 = remover.unitary
    print(circuit1)
    print("The old circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit1, u1, noise, key))

    remover.reduce_circuit()

    circuit2 = remover.circuit
    print(circuit2)
    print("I reduced the gate count by", len(circuit1) - len(circuit2), "out of", len(circuit1))
    u2 = remover.unitary
    print("The new circuit run with noise has this fidelity to the target unitary", f_pro_experimental(circuit2, u1, noise, key))
    print("The fidelity between the ideal circuits is", f_pro([u2], u1))


def run_multiple():
    circuits = [random.choices(list(key.keys()), k=20, weights=weights) for _ in range(10)]
    run_all(circuits, key, noise, n_qubits, True)


def synthesise(u: np.ndarray):
    circuit1 = random.choices(list(key.keys()), k=20, weights=weights)
    remover = SubcircuitRemover(circuit1, key, noise, n_qubits=n_qubits, verbose=True)
    remover.set_target_unitary(u)
    print("The random circuit and random unitary have this fidelity", f_pro_experimental(circuit1, u, [], key))
    while remover.replace_any_subcircuit():
        pass
    print("The new circuit has this fidelity", f_pro_experimental(remover.circuit, u, [], key))
    return remover.circuit


def synth_n_qubits():
    u = reduce(lambda x, y: x @ y, random.choices(list(key.values()), k=10, weights=weights), np.eye(2 ** n_qubits))
    print("synthesising\n", u)
    return synthesise(u)


run_multiple()

