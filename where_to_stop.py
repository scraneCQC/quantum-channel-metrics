import approximations
import J_fidelity
import S_fidelity
import diamond_distance
import numpy as np
import math
import matplotlib.pyplot as plt
import time
from scipy.optimize import curve_fit


def plot_distances(circuits, U, noise_strength):
    max_acc = len(circuits)

    p1 = noise_strength
    gamma1 = noise_strength
    gamma2 = noise_strength

    start = time.time()
    J_fidelities = [(2 - 2 * J_fidelity.f_pro_experimental(c, U, p1, gamma1, gamma2) ** 0.5) ** 0.5 for c in circuits]
    end = time.time()
    print("The best accuracy for J was: " + str(min(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    #J_noiseless = [(2 - 2 * J_fidelity.f_pro_experimental(c, U) ** 0.5) ** 0.5 for c in circuits]

    # This may take a few seconds
    start = time.time()
    S_fidelities = [(2 - 2 * S_fidelity.experimental(c, U, 100, p1=p1, gamma1=gamma1, gamma2=gamma2) ** 0.5) ** 0.5
                    for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(min(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print()

    start = time.time()
    diamond_distances = [diamond_distance.monte_carlo_from_circuit(c, U, 1000, p1, gamma1, gamma2) for c in circuits]
    end = time.time()
    print("The best accuracy for diamond was: " + str(min(range(max_acc), key=lambda x: diamond_distances[x])))
    print("It took " + str(end - start) + " seconds")
    print()

    fig, ax1 = plt.subplots()
    lineJ, = ax1.plot(J_fidelities)
    lineS, = ax1.plot(S_fidelities)
    lineD, = ax1.plot(diamond_distances)

    ax1.set_ylabel('distance', color='tab:blue')
    ax1.set_xlabel('accuracy')
    ax1.legend((lineJ, lineS, lineD), ("J_fidelity", "S_fidelity", "diamond"))
    plt.savefig("dist.png")


def plot_fidelities(circuits, U, noise_strength):
    max_acc = len(circuits)

    lengths = [len(c) for c in circuits]

    p1 = noise_strength
    gamma1 = noise_strength
    gamma2 = noise_strength

    J_noiseless = [J_fidelity.f_pro_experimental(c, U) for c in circuits]

    def model(n, a, b):
        return 1 - (a * 0.5 ** n) + b * noise_strength * n

    start = time.time()
    J_fidelities = [J_fidelity.f_pro_experimental(c, U, p1, gamma1, gamma2) for c in circuits]
    end = time.time()
    print("The best accuracy for J was: " + str(min(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end - start) + " seconds")
    params1 = curve_fit(model, lengths, J_fidelities)[0]
    print("gradient for J is "+str(params1[1]))
    print()

    S_fidelities = [S_fidelity.experimental(c, U, 100, p1=p1, gamma1=gamma1, gamma2=gamma2) for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(min(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end - start) + " seconds")
    params2 = curve_fit(model, lengths, S_fidelities)[0]
    print("gradient for S is " + str(params2[1]))
    print()

    # This will take several minutes and is really really inaccurate with any amount of noise
    # start = time.time()
    # J_simulated = [J_fidelity.f_pro_simulated(c, U, p1, gamma1, gamma2) for c in circuits]
    # end = time.time()
    # print("The best accuracy for simulation was: " + str(max(range(max_acc), key=lambda x: J_simulated[x])))
    # print("It took " + str(end-start) + " seconds")

    plt.figure()
    lineJ2, = plt.plot(lengths, J_noiseless)

    lineJ, = plt.plot(lengths, J_fidelities)

    lineS, = plt.plot(lengths, S_fidelities)

    plt.xlabel('Circuit length')
    plt.ylabel('Fidelity')
    plt.legend((lineJ2, lineJ, lineS), ('J_fidelity (noiseless)', 'J_fidelity', 'S_fidelity'))
    plt.savefig('fid.png')
    return


max_acc = 20


theta = math.pi/3


circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)
# circuits = ["HH"*i for i in range(max_acc)]

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])
# U = np.eye(2)

plot_distances(circuits, U, 1e-4)
# plot_fidelities(circuits, U, 1e-4)

