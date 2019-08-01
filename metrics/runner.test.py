from metrics.approximation_runner import *
import time

s = "HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT"

#for m in one_qubit_diracs:
#    print(get_expectation(I1, s) - np.trace(I1 @ density_runner.run_by_matrices(s, np.array([[1, 0], [0, 0]]))).real)

circuit = Circuit(1)
circuit.Rx(0, -0.5)

start = time.time()
print(get_pauli_expectation_v2("X", circuit, "Y", shots=1000000))
end = time.time()
print(end-start)
