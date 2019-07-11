import subprocess


def generate_circuit(theta, accuracy):
    res = subprocess.run(["../gridsynth", theta, "-d "+str(accuracy), "-p", "-r 0"],
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return str(res.stdout)[2:-3]


def save_circuits(theta, out=None, max_accuracy=10):
    if out is None:
        out = theta + ".txt"
    file = open(out, "w+")
    for i in range(max_accuracy):
        file.write(generate_circuit(theta, i+1)+"\n")
    file.close()
