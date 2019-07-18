import approximations
import J_fidelity
import J_distance
import S_fidelity
import diamond_distance
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit
from typing import Any, Iterable
from noise import standard_noise_channels


def plot_distances(circuits: Iterable[Iterable[Any]], U: np.ndarray, noise_strength: float):
    max_acc = len(circuits)

    noise_channels = standard_noise_channels(noise_strength)

    start = time.time()
    J_fidelities = [(2 - 2 * J_fidelity.f_pro_experimental(c, U, noise_channels) ** 0.5) ** 0.5 for c in circuits]
    end = time.time()
    print("The best accuracy for J fidelity was: " + str(min(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    start = time.time()
    J_distances = [J_distance.j_distance_experimental(c, U, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for J distance was: " + str(min(range(max_acc), key=lambda x: J_distances[x])))
    print("It took " + str(end-start) + " seconds")
    print()


    # This may take a few seconds
    start = time.time()
    S_fidelities = [(2 - 2 * S_fidelity.experimental(c, U, 100, noise_channels) ** 0.5) ** 0.5
                    for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(min(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    start = time.time()
    diamond_distances = [diamond_distance.monte_carlo_from_circuit(c, U, 1000, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for diamond was: " + str(min(range(max_acc), key=lambda x: diamond_distances[x])))
    print("It took " + str(end - start) + " seconds")
    print()

    fig, ax1 = plt.subplots()
    line_jf, = ax1.plot(J_fidelities)
    line_jd, = ax1.plot(J_distances)
    lineS, = ax1.plot(S_fidelities)
    lineD, = ax1.plot(diamond_distances)

    ax1.set_ylabel('distance', color='tab:blue')
    ax1.set_xlabel('accuracy')
    ax1.legend((line_jf, line_jd, lineS, lineD), ("J_fidelity", "J distance", "S_fidelity", "diamond"))
    plt.savefig("graphs/dist.png")


def plot_fidelities(circuits: Iterable[Iterable[Any]], U: np.ndarray, noise_strength: float):
    max_acc = len(circuits)

    lengths = [len(c) for c in circuits]

    noise_channels = standard_noise_channels(noise_strength)

    J_noiseless = [J_fidelity.f_pro_experimental(c, U) for c in circuits]

    def model(n, a, b):
        return 1 - (a * 0.5 ** n) + b * noise_strength * n

    start = time.time()
    J_fidelities = [J_fidelity.f_pro_experimental(c, U, noise_channels) for c in circuits]
    end = time.time()
    print("The best accuracy for J was: " + str(max(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end - start) + " seconds")
    params1 = curve_fit(model, lengths, J_fidelities)[0]
    print("gradient for J is "+str(params1[1]))
    print()

    start=time.time()
    S_fidelities = [S_fidelity.experimental(c, U, 100, noise_channels) for c in circuits]
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
    lineJ2, = plt.plot( J_noiseless)
    lineJ, = plt.plot(J_fidelities)
    lineS, = plt.plot(S_fidelities)

    plt.xlabel('Accuracy')
    plt.ylabel('Fidelity')
    plt.legend((lineJ2, lineJ, lineS), ('J_fidelity (noiseless)', 'J_fidelity', 'S_fidelity'))
    plt.savefig('graphs/fid.png')
    return


max_acc = 20


theta = math.pi/3


circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)
# circuits = ["HH"*i for i in range(max_acc)]

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])
# U = np.eye(2)

# plot_distances(circuits, U, 1e-3)
plot_fidelities(circuits, U, 1e-4)

