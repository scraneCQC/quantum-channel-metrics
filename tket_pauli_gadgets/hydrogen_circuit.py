from tket_pauli_gadgets.tket_circuit_rewriter import RewriteTket
from tket_pauli_gadgets.noise_models import channels
from chem import circ

# based on ibmqx4 averages
single_noise, cnot_noise = channels(amplification=20)

rewriter = RewriteTket(circ, single_noise, cnot_noise, verbose=True)

print(rewriter.fidelity(circ.get_commands()))

truncated = rewriter.instructions_to_circuit(circ.get_commands()[:24])

rewriter.set_circuit_and_target(truncated)

print(rewriter.fidelity(truncated.get_commands()))

