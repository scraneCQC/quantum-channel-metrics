from qiskit import IBMQ
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import amplitude_damping_error, depolarizing_error, phase_damping_error
import numpy as np
import noise


def amplified_qiskit_model(name, amplification=1, gate_time=0.001):
    provider = IBMQ.load_account()
    device = provider.get_backend(name)
    properties = device.properties()
    noise_model = NoiseModel()

    # depolarizing error
    cnot_depol_errs = [x.parameters[0].value for x in properties.gates if x.gate == 'cx']
    ave_cnot_error = sum(cnot_depol_errs) / len(cnot_depol_errs) / amplification
    cnot_depol_error = depolarizing_error(ave_cnot_error, 2)

    single_depol_errs = [x.parameters[0].value for x in properties.gates if x.gate != 'cx']
    ave_single_error = sum(single_depol_errs) / len(single_depol_errs) / amplification
    single_depol_error = depolarizing_error(ave_single_error, 1)

    # amp damp
    t1s = [x[0].value for x in properties.qubits]
    ave_t1 = sum(t1s) / len(t1s)
    t1_ratio = gate_time / (ave_t1 * amplification)
    amp_gamma = 1 - np.exp(-t1_ratio)
    amp_error = amplitude_damping_error(amp_gamma)

    # phase damp
    t2s = [x[1].value for x in properties.qubits]
    ave_t2 = sum(t2s) / len(t2s)
    t2_ratio = gate_time / (ave_t2 * amplification)
    phase_gamma = 1 - np.exp(-t2_ratio)
    phase_error = phase_damping_error(phase_gamma)

    thermal_error = amp_error.compose(phase_error)
    noise_model.add_all_qubit_quantum_error(single_depol_error.compose(thermal_error), ['u1', 'u2', 'u3'])

    noise_model.add_all_qubit_quantum_error(cnot_depol_error.compose(thermal_error.tensor(thermal_error)), ['cx'])

    return noise_model


def channels(amplification=1, gate_time=0.001):
    provider = IBMQ.load_account()
    device = provider.get_backend("ibmqx4")
    properties = device.properties()

    # depolarizing error
    cnot_depol_errs = [x.parameters[0].value for x in properties.gates if x.gate == 'cx']
    ave_cnot_error = sum(cnot_depol_errs) / len(cnot_depol_errs) / amplification

    single_depol_errs = [x.parameters[0].value for x in properties.gates if x.gate != 'cx']
    ave_single_error = sum(single_depol_errs) / len(single_depol_errs) / amplification

    # amp damp
    t1s = [x[0].value for x in properties.qubits]
    ave_t1 = sum(t1s) / len(t1s)
    t1_ratio = gate_time / (ave_t1 * amplification)
    amp_gamma = 1 - np.exp(-t1_ratio)

    # phase damp
    t2s = [x[1].value for x in properties.qubits]
    ave_t2 = sum(t2s) / len(t2s)
    t2_ratio = gate_time / (ave_t2 * amplification)
    phase_gamma = 1 - np.exp(-t2_ratio)

    singles_channel = noise.channels(ave_single_error, amp_gamma, phase_gamma, 1)
    cnot_channel = noise.channels(ave_cnot_error, amp_gamma, phase_gamma, 2)

    return singles_channel, cnot_channel

