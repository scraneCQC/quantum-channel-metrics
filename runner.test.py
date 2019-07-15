from approximation_runner import *
import density_runner
from Pauli import *

s = "HTSHTHTSHTSHTHTSHTHTHTSHTSHTHTSHTSHTSHTSHTSHTHTSHTSHTHT"

#for m in one_qubit_diracs:
#    print(get_expectation(I1, s) - np.trace(I1 @ density_runner.run_by_matrices(s, np.array([[1, 0], [0, 0]]))).real)


def my_circuit_maker(c):
    c.measure_all()
    c.H(0)
    c.measure_all()
print(run_circuit("I", 0,0,0, prep_circuit=my_circuit_maker))