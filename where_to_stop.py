import approximations
import J_fidelity
import S_fidelity
import numpy as np
import math
import matplotlib.pyplot as plt
import time


def run(circuits, U, noise_strength, plot_against_length=False):
    max_acc = len(circuits)

    p1 = noise_strength
    gamma1 = noise_strength
    gamma2 = noise_strength

    start = time.time()
    J_fidelities = [J_fidelity.f_pro_experimental(c, U, p1, gamma1, gamma2) for c in circuits]
    end = time.time()
    print("The best accuracy for J was: " + str(max(range(max_acc), key=lambda x: J_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print("gradient for J is "+str((J_fidelities[-1] - J_fidelities[0])/max_acc))
    print()

    J_noiseless = [J_fidelity.f_pro_experimental(c, U) for c in circuits]

    # This may take a few seconds
    start = time.time()
    S_fidelities = [S_fidelity.experimental(c, U, 100, p1=p1, gamma1=gamma1, gamma2=gamma2) for c in circuits]
    end = time.time()
    print("The best accuracy for S was: " + str(max(range(max_acc), key=lambda x: S_fidelities[x])))
    print("It took " + str(end-start) + " seconds")
    print("gradient for S is " + str((S_fidelities[-1] - S_fidelities[0]) / max_acc))
    print()

    # This will take several minutes and is really really inaccurate with any amount of noise
    # start = time.time()
    # J_simulated = [J_fidelity.f_pro_simulated(c, U, p1, gamma1, gamma2) for c in circuits]
    # end = time.time()
    # print("The best accuracy for simulation was: " + str(max(range(max_acc), key=lambda x: J_simulated[x])))
    # print("It took " + str(end-start) + " seconds")

    if plot_against_length:
        lengths = [len(c) for c in circuits]
        plt.figure()
        lineJ, = plt.plot(lengths, J_fidelities)
        lineS, = plt.plot(lengths, S_fidelities)
        line_model1, = plt.plot(lengths, [1 - (0.07 * (2 ** -(n+1)) + 1.7 * noise_strength * lengths[n]) for n in range(max_acc)])
        line_model2, = plt.plot(lengths, [1 - (0.062 * (2 ** -(n+1)) + 4.5 * noise_strength * lengths[n]) for n in range(max_acc)])
        plt.xlabel('Circuit length')
        plt.ylabel('Fidelity')
        plt.legend((lineJ, lineS, line_model1, line_model2), ('J_fidelity', 'S_fidelity', 'model', 'model 2'))
        plt.savefig('out.png')
        return

    fig, ax1 = plt.subplots()
    lineJ, = ax1.plot(J_fidelities)
    lineS, = ax1.plot(S_fidelities)

    #lineJ_sim, = ax1.plot(J_simulated)
    lineJ2, = ax1.plot(J_noiseless)
    ax1.set_ylabel('fidelity', color='tab:blue')
    ax1.set_xlabel('accuracy')
    ax2 = ax1.twinx()
    line_length, = ax2.plot([len(c) for c in circuits], color='tab:red')
    ax2.set_ylabel('circuit length', color='tab:red')
    ax1.legend((lineJ, lineJ2, lineS, line_length), ("J_fidelity (noisy)", "J_fidelity (noiseless)", "S_fidelity (noisy)", "Circuit length"))
    plt.savefig("out.png")


max_acc = 20


theta = math.pi/3


circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)
# circuits = ["HH"*i for i in range(max_acc)]

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])
# U = np.eye(2)

run(circuits, U, 1e-4, True)
