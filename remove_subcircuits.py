from J_fidelity import f_pro, f_pro_experimental
from functools import reduce
import numpy as np


class SubcircuitRemover:

    def __init__(self, circuit_description, circuit_key, noise_channels):
        self.circuit = circuit_description
        self.key = circuit_key
        self.noise_channels = noise_channels
        self.original_unitary = self.unitary

    def should_remove_subcircuit(self, start, end):
        subcircuit = self.circuit[start:end]
        unitary = reduce(lambda x, y: x @ y, [self.key[s] for s in subcircuit], np.eye(2))
        do_it = f_pro_experimental(subcircuit, unitary, self.noise_channels, self.key)
        dont_do_it = f_pro([unitary], np.eye(unitary.shape[0]))
        return do_it <= dont_do_it

    def test_all_subcircuits(self):
        length = len(self.circuit)
        for i in range(length, 0, -1):  # Test subcircuits of length i
            for s in range(length - i + 1):  # That start at s
                e = s + i
                res = self.should_remove_subcircuit(s, s + i)
                if res:
                    self.circuit = self.circuit[:s] + self.circuit[e:]
                    return True
        return False

    def reduce_circuit(self):
        while self.test_all_subcircuits():
            pass

    @property
    def unitary(self):
        return reduce(lambda x, y: x @ y, [self.key[s] for s in self.circuit], np.eye(2))