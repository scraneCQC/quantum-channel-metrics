import subprocess
from typing import List


def generate_circuit(theta: str, accuracy: int) -> str:
    res = subprocess.run(["../../gridsynth", theta, "-b "+str(accuracy), "-p", "-r 0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(res.stdout)[2:-3]


def save_circuits(theta: str, out: str = None, max_accuracy: int = 10):
    if out is None:
        out = theta + ".txt"
    file = open(out, "w+")
    for i in range(max_accuracy):
        file.write(generate_circuit(theta, i+1)+"\n")
    file.close()


def get_circuits(theta: str, max_accuracy:int = 10) -> List[str]:
    return [generate_circuit(theta, a+1) for a in range(max_accuracy)]

