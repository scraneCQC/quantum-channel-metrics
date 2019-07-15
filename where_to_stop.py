import approximations
import J_fidelity
import S_fidelity
import numpy as np
import math
import matplotlib.pyplot as plt
import time


theta = math.pi/3

max_acc = 15
circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)

p1 = 0.00003
gamma1 = 0.00003
gamma2 = 0.00003

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])

start = time.time()
J_fidelities = [J_fidelity.f_pro_experimental(c, U, p1, gamma1, gamma2) for c in circuits]
end = time.time()
print("The best accuracy for J was: " + str(max(range(max_acc), key=lambda x: J_fidelities[x])))
print("It took " + str(end-start) + " seconds")


# This may take a few seconds
start = time.time()
S_fidelities = [S_fidelity.experimental(c, U, 100, p1=p1, gamma1=gamma1, gamma2=gamma2) for c in circuits]
end = time.time()
print("The best accuracy for S was: " + str(max(range(max_acc), key=lambda x: S_fidelities[x])))
print("It took " + str(end-start) + " seconds")

# This will take several minutes
start = time.time()
J_simulated = [J_fidelity.f_pro_simulated(c, U, p1, gamma1, gamma2) for c in circuits]
end = time.time()
print("The best accuracy for simulation was: " + str(max(range(max_acc), key=lambda x: J_simulated[x])))
print("It took " + str(end-start) + " seconds")

fig, ax1 = plt.subplots()
lineJ, = ax1.plot(J_fidelities)
lineS, = ax1.plot(S_fidelities)
lineJ_sim, = ax1.plot(J_simulated)
ax1.set_ylabel('fidelity', color='tab:blue')
ax1.set_xlabel('accuracy')
ax2 = ax1.twinx()
line_length, = ax2.plot([len(c) for c in circuits], color='tab:red')
ax2.set_ylabel('circuit length', color='tab:red')
ax1.legend((lineJ, lineS, lineJ_sim, line_length), ("J_fidelity (theoretical)", "S_fidelity (theoretical)", "J_fidelity (simulated)", "Circuit length"))
fig.tight_layout()
plt.savefig("out.png")