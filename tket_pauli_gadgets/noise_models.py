from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import amplitude_damping_error, depolarizing_error, phase_damping_error
import numpy as np


def amplified_qiskit_model(name, amplification=1, gate_time=0.001):
    device = IBMQ.get_backend(name)
    coupling = device.configuration().coupling_map
    properties = device.properties()
    noise_model = NoiseModel()

    # depolarizing error
    cnot_errs = [x.parameters[0].value for x in properties.gates if x.gate == 'cx']
    ave_cnot_error = sum(cnot_errs) / len(cnot_errs)
    print(ave_cnot_error)
    noise_model.add_all_qubit_quantum_error(depolarizing_error(ave_cnot_error / amplification, 2), ['cx'])

    # amp damp
    t1s = [x[0] for x in properties.qubits]
    ave_t1 = sum(t1s) / len(t1s)
    t1_ratio = gate_time / (ave_t1 * amplification)
    amp_gamma = 1 - np.exp(-t1_ratio)
    print(amp_gamma)
    amp_error = amplitude_damping_error(amp_gamma)
    noise_model.add_all_qubit_quantum_error(amp_error, ['u2', 'u3'])

    noise_model.add_all_qubit_quantum_error(amp_error.tensor(amp_error), ['cx'])

    # phase damp
    t2s = [x[1] for x in properties.qubits]
    ave_t2 = sum(t2s) / len(t2s)
    t2_ratio = gate_time / (ave_t2 * amplification)
    phase_gamma = 1 - np.exp(-t2_ratio)
    print(phase_gamma)
    phase_error = phase_damping_error(phase_gamma)
    noise_model.add_quantum_error(phase_error, ['u2', 'u3'])

    noise_model.add_all_qubit_quantum_error(phase_error.tensor(phase_error), ['cx'])

    return noise_model

amplified_qiskit_model('ibmqx4')