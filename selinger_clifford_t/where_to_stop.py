from selinger_clifford_t import approximations
from metrics import J_distance, J_fidelity, S_fidelity, diamond_distance
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from typing import Any, Iterable
from noise import standard_noise_channels
from tket_pauli_gadgets.noise_models import channels


def plot_distances(circuits: Iterable[Iterable[Any]], unitary: np.ndarray, noise_channels: Iterable[Iterable[np.ndarray]]):
    max_acc = len(circuits)

    pro_fid = J_fidelity.ProcessFidelityFinder(1)
    start = time.time()
    J_fidelities = [(1 - pro_fid.f_pro_experimental(c, unitary, noise_channels) ** 0.5) ** 0.5 for c in circuits]
    end = time.time()
    print("The best accuracy for J fidelity was: " + str(min(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    start = time.time()
    J_distances = [J_distance.j_distance_experimental(c, unitary, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for J distance was: " + str(min(range(max_acc), key=lambda x: J_distances[x])))
    print("It took " + str(end-start) + " seconds")
    print()


    # This may take a few seconds
    start = time.time()
    S_fidelities = [(1 - S_fidelity.experimental(c, unitary, 1000, noise_channels) ** 0.5) ** 0.5
                    for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(min(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    start = time.time()
    diamond_distances = [diamond_distance.monte_carlo_from_circuit(c, unitary, 100, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for diamond was: " + str(min(range(max_acc), key=lambda x: diamond_distances[x])))
    print("It took " + str(end - start) + " seconds")
    print()

    fig, ax1 = plt.subplots()
    line_jf, = ax1.plot(J_fidelities)
    line_jd, = ax1.plot(J_distances)
    lineS, = ax1.plot(S_fidelities)
    lineD, = ax1.plot(diamond_distances)

    ax1.set_ylabel('distance')
    ax1.set_xlabel('accuracy')
    ax1.legend((line_jf, line_jd, lineS, lineD), ("process fidelity", "process distance", "stabilised fidelity", "diamond distance"))
    plt.savefig("../graphs/dist_ibm_noise.png")


def plot_fidelities(circuits: Iterable[Iterable[Any]], U: np.ndarray, noise_strength: float):
    max_acc = len(circuits)

    lengths = [len(c) for c in circuits]

    noise_channels = standard_noise_channels(noise_strength)

    pro_fid = J_fidelity.ProcessFidelityFinder(1)
    J_noiseless = [pro_fid.f_pro_experimental(c, U) for c in circuits]

    def model(n, a, b):
        return 1 - (a * 0.5 ** n) + b * noise_strength * n

    start = time.time()
    J_fidelities = [pro_fid.f_pro_experimental(c, U, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for J was: " + str(max(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end - start) + " seconds")
    params1 = curve_fit(model, lengths, J_fidelities)[0]
    print("gradient for J is "+str(params1[1]))
    print()

    start=time.time()
    S_fidelities = [S_fidelity.experimental(c, U, 1000, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(max(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end - start) + " seconds")
    params2 = curve_fit(model, lengths, S_fidelities)[0]
    print("gradient for S is " + str(params2[1]))
    print()

    # This will take several minutes and is really really inaccurate with any amount of noise
    # start = time.time()
    # J_simulated = [J_fidelity.f_pro_simulated(c, U, noise_strength, noise_strength, noise_strength) for c in circuits]
    # end = time.time()
    # print("The best accuracy for simulation was: " + str(max(range(max_acc), key=lambda x: J_simulated[x])))
    # print("It took " + str(end-start) + " seconds")

    plt.figure()
    lineJ2, = plt.plot(J_noiseless)
    lineJ, = plt.plot(J_fidelities)
    lineS, = plt.plot(S_fidelities)

    plt.xlabel('Accuracy')
    plt.ylabel('Fidelity')
    plt.legend((lineJ2, lineJ, lineS), ('J_fidelity (noiseless)', 'J_fidelity', 'S_fidelity'))
    plt.savefig('../graphs/fid.png')
    return


def plot_best_vs_noise(circuits, unitary, noises):

    pro_fid = J_fidelity.ProcessFidelityFinder(1)
    start = time.time()
    best_j_fids = [min([(1 - pro_fid.f_pro_experimental(c, unitary, n) ** 0.5) ** 0.5 for c in circuits]) for n in noises]
    end = time.time()
    print("J fidelity took " + str(end - start) + " seconds")
    print()

    start = time.time()
    best_j_dist = [min([J_distance.j_distance_experimental(c, unitary, n) for c in circuits]) for n in noises]
    end = time.time()
    print("J distance " + str(end - start) + " seconds")
    print()

    # This may take a few seconds
    start = time.time()
    best_s_fids = [min([(1 - S_fidelity.experimental(c, unitary, 100, n) ** 0.5) ** 0.5 for c in circuits])
                   for n in noises]
    end = time.time()
    print("S fid took " + str(end - start) + " seconds")
    print()

    start = time.time()
    best_diamond_dist = [min([diamond_distance.monte_carlo_from_circuit(c, unitary, 100, n) for c in circuits]) for n in noises]
    end = time.time()
    print("Diamond took " + str(end - start) + " seconds")
    print()

    amps = [a + 1 for a in range(10)]
    plt.figure()
    plt.plot(amps, best_j_fids, label="process fidelity")
    plt.plot(amps, best_j_dist, label="process distance")
    plt.plot(amps, best_s_fids, label="stabilised fidelity")
    plt.plot(amps, best_diamond_dist, label="diamond distance")
    plt.legend()

    plt.xlabel('gate time reduction')
    plt.ylabel('best distance')

    plt.savefig("../graphs/best_ibm_noise.png")

amplifications = list(range(10))
amplification = 1
single_noise, cnot_noise = channels(amplification=amplification)
noises = [channels(amplification=a+1)[0] for a in amplifications]

max_acc = 20
theta = math.pi/3
circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])

# plot_distances(circuits, U, single_noise)
# plot_fidelities(circuits, U, 1e-4)
plot_best_vs_noise(circuits, U, noises)
