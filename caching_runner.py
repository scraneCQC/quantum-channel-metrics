from Pauli import get_diracs
import numpy as np
from functools import reduce
from typing import Dict, Iterable


class CachingRunner:

    def __init__(self, key: Dict, n_qubits: int, noise_channels: Iterable):
        self.n_qubits = n_qubits
        self.diracs = get_diracs(n_qubits)
        self.cache = {k: reduce(lambda x, y: x @ y, (self.process_matrix(channel) for channel in noise_channels), self.process_matrix([v]))
                      for k, v in key.items()}
        self.I = np.eye(2 ** (2 * n_qubits))
        self.d2 = 2 ** (2 * n_qubits)

    def remember(self, circuit):
        if "".join(circuit) in self.cache:
            return
        self.cache.update({"".join(circuit): self.get_matrix(circuit)})

    def get_matrix(self, circuit):
        if "".join(circuit) in self.cache:
            return self.cache["".join(circuit)]
        l = len(circuit)
        for i in range(len(circuit)):
            if "".join(circuit[i:]) in self.cache:
                return self.get_matrix(circuit[:i]) @ self.cache["".join(circuit[i:])]
            if "".join(circuit[:(l - i)]) in self.cache:
                return self.cache["".join(circuit[:(l - i)])] @ self.get_matrix(circuit[(l - i):])
        return reduce(lambda x, y: x @ y, [self.cache[s] for s in circuit], self.I)

    def decompose(self, density):
        return [np.einsum('ij,ji->', density, d) / (2 ** self.n_qubits) for d in self.diracs]

    def process_matrix(self, channel):
        return np.vstack([self.decompose(sum(e @ d @ e.transpose().conjugate() for e in channel))
                          for d in self.diracs]).transpose()

    def circuit_process_matrix(self, circuit):
        return reduce(lambda x, y: x @ y, (self.cache[g] for g in circuit), self.I)

    def to_density(self, coefficients):
        return sum([r * d for r, d in zip(coefficients, self.diracs)])

    def run(self, circuit, density):
        start = np.array([[r] for r in self.decompose(density)])
        if "".join(circuit) in self.cache:
            return self.to_density(self.cache["".join(circuit)] @ start)
        mat = self.get_matrix(circuit)
        self.cache.update({"".join(circuit): mat})
        return self.to_density(mat @ start)
