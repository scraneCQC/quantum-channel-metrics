from metrics.density_runner import run_by_matrices, ops
from caching_runner import CachingRunner
from Pauli import one_qubit_diracs
import cmath


noise_channels = []  # standard_noise_channels(0.01)

cr = CachingRunner(ops, 1, noise_channels)

test_circuits = ["S", "X", "H", "T", "SXHT"]

for c in test_circuits:
    for rho in one_qubit_diracs:
        a = run_by_matrices(c, rho, noise_channels)
        b = cr.run(c, rho)
        if not all(cmath.isclose(a[i][j], b[i][j], abs_tol=1e-9) for i in range(2) for j in range(2)):
            print("CR failed to run", c, "on\n", rho, "\ngave\n", b, "\nbut should have given\n", a)
        else:
            print("Success!")
        print()
