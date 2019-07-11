import approximations
import J_fidelity
import numpy as np
import math
import matplotlib.pyplot as plt


theta = math.pi/3

max_acc = 20
circuits = approximations.get_circuits(str(theta), max_accuracy=max_acc)

p1 = 0.0001
gamma1 = 0.0001
gamma2 = 0

U = np.array([[complex(math.cos(theta / 2), - math.sin(theta / 2)), 0],
              [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])

fidelities = [J_fidelity.f_pro_experimental(c, U, p1, gamma1, gamma2) for c in circuits]
print("The best accuracy was: "+str(max(range(max_acc), key=lambda x: fidelities[x])))


fig, ax1 = plt.subplots()
ax1.plot(fidelities)
ax1.set_ylabel('fidelity', color='tab:blue')
ax1.set_xlabel('accuracy')
ax2 = ax1.twinx()
ax2.plot([len(c) for c in circuits], color='tab:red')
ax2.set_ylabel('circuit length', color='tab:red')
fig.tight_layout()
plt.savefig("out.png")