from approximation_runner import *
import density_runner
from Pauli import *
import time

s = "HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT"

#for m in one_qubit_diracs:
#    print(get_expectation(I1, s) - np.trace(I1 @ density_runner.run_by_matrices(s, np.array([[1, 0], [0, 0]]))).real)

circuit = Circuit(1)
circuit.Rx(0, -0.5)

print(circuit.get_commands())

start = time.time()
print(get_pauli_expectation("X", circuit, "Y", 1, shots=100))
end = time.time()
# print(end-start)
