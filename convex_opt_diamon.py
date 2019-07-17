# Import packages.
import cvxpy as cp
import numpy as np
from noise import amplitude_damping_channel

n=2

c = amplitude_damping_channel(0.1)
A = np.kron(c[0], np.array([[1], [0]])) + np.kron(c[1], np.array([[0], [1]]))
B = A

A = np.random.random((4, 2))
B = np.random.random((4, 2))

inner = B @ B.transpose().conjugate()


def partial_trace_y(M):
    pt1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0]]) @ M @ np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
    pt2 = np.array([[0, 0, 1, 0], [0, 0, 1, 0]]) @ M @ np.array([[0, 0], [0, 0], [1, 0], [0, 1]])
    return pt1 + pt2


constraints = []
# Define and solve the CVXPY problem.
# Create a symmetric matrix variable.

# X = X1 + iX2 is Hermitian
X1 = cp.Variable((n, n), PSD=True)
#X2 = cp.Variable((n, n))
#constraints += [X2.T == - X2]

# W = W1 + iW2 is Hermitian
W1 = cp.Variable((n**2, n**2), PSD=True)
#W2 = cp.Variable((n**2, n**2))
#constraints += [W2.T == W2]


def check_traces(w1, w2, x1, x2, A):
    lhs = partial_trace_y(w1 + complex(0,1) * w2)
    rhs = partial_trace_y(A @ (x1 + complex(0,1) * x2) @ A.transpose().conjugate())
    return lhs == rhs


constraints += [cp.trace(X1) <= 1]
constraints += [check_traces(W1, 0, X1, 0, A)]  # TODO

prob = cp.Problem(cp.Maximize(cp.trace(inner @ W1)), constraints)
prob.solve(verbose=True, solver=cp.CVXOPT)
print(prob.status)

if prob.status == "optimal":
    # Print result.
    print("The optimal value is", prob.value)
    print("A solution X is")
    print(X1.value)
    print("W is")
    print(W1.value)
