from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import amplitude_damping_error, depolarizing_error, phase_damping_error
import numpy as np


def amplified_qiskit_model(name, amplification=1, gate_time=0.001):
    device = IBMQ.get_backend(name)
    coupling = device.configuration().coupling_map
    properties = device.properties()
    noise_model = NoiseModel()
    #   add multi qubit depolarizing error
    for i in range(0, len(properties.gates)):
        if len(properties.gates[i].qubits) == 2:
            depol_error = depolarizing_error(properties.gates[i].parameters[0].value / amplification, 2)
            noise_model.add_quantum_error(depol_error, properties.gates[i].gate, properties.gates[i].qubits)
    #   add single qubit amplitude damping, phase damping and readout errors
    for i in range(0, len(properties.qubits)):
        # amp damp
        t1_ratio = gate_time / (properties.qubits[i][0].value * amplification)
        amp_gamma = 1 - np.exp(-t1_ratio)
        amp_error = amplitude_damping_error(amp_gamma)
        noise_model.add_quantum_error(amp_error, ['u2', 'u3'], [i])
        # phase damp
        t2_ratio = gate_time / (properties.qubits[i][1].value * amplification)
        phase_gamma = 1 - np.exp(-t2_ratio)
        phase_error = phase_damping_error(phase_gamma)
        noise_model.add_quantum_error(phase_error, ['u2', 'u3'], [i])
    # add multi qubit amp damp and phase damp
    for i in range(0, len(coupling)):
        t1_1 = properties.qubits[coupling[i][0]][0].value * amplification
        t1_2 = properties.qubits[coupling[i][1]][0].value * amplification
        t2_1 = properties.qubits[coupling[i][0]][1].value * amplification
        t2_2 = properties.qubits[coupling[i][1]][1].value * amplification

        t1_ratio_1 = gate_time / t1_1
        t1_ratio_2 = gate_time / t1_2
        t2_ratio_1 = gate_time / t2_1
        t2_ratio_2 = gate_time / t2_2

        amp_gamma_1 = 1 - np.exp(-t1_ratio_1)
        amp_gamma_2 = 1 - np.exp(-t1_ratio_2)
        phase_gamma_1 = 1 - np.exp(-t2_ratio_1)
        phase_gamma_2 = 1 - np.exp(-t2_ratio_2)

        amp_error_1 = amplitude_damping_error(amp_gamma_1)
        amp_error_2 = amplitude_damping_error(amp_gamma_2)
        amp_error_mq = amp_error_1.tensor(amp_error_2)

        phase_error_1 = phase_damping_error(phase_gamma_1)
        phase_error_2 = phase_damping_error(phase_gamma_2)
        phase_error_mq = phase_error_1.tensor(phase_error_2)

        noise_model.add_quantum_error(amp_error_mq, ['cx'], coupling[i])
        noise_model.add_quantum_error(phase_error_mq, ['cx'], coupling[i])
    return noise_model
