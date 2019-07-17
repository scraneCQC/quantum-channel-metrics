import numpy as np
import picos as pic
from noise import amplitude_damping_channel


def solve_with_pic(A, B):
    L1 = pic.new_param('L1', np.array([[1, 0, 0, 0], [0, 1, 0, 0]]))
    R1 = pic.new_param('R1', np.array([[1, 0], [0, 1], [0, 0], [0, 0]]))
    L2 = pic.new_param('L2', np.array([[0, 0, 1, 0], [0, 0, 1, 0]]))
    R2 = pic.new_param('R2', np.array([[0, 0], [0, 0], [1, 0], [0, 1]]))

    def partial_trace_y(M):
        pt1 = L1 * M * R1
        pt2 = L2 * M * R2
        return pt1 + pt2

    A = pic.new_param('A', A)
    B = pic.new_param('B', B)
    dim_x, dim_y, dim_z = 2, 2, 2
    F = pic.Problem()
    X = F.add_variable('X', (dim_x, dim_x), 'hermitian')
    W = F.add_variable('W', (dim_y * dim_z, dim_y * dim_z), 'hermitian')
    F.set_objective('max', B * B.transpose().conjugate() | W)
    F.add_constraint('I'|X.real < 1)  # Trace at most 1
    F.add_constraint(X >> 0)
    F.add_constraint(W >> 0)
    # TODO add the constraint about partial traces over Y
    F.add_constraint(partial_trace_y(W) << partial_trace_y(A * X * A.conjugate().transpose()))
    print(F)
    F.solve(verbose=0, solver='cvxopt')
    print(F.obj_value())


c = amplitude_damping_channel(0)
A = np.kron(c[0], np.array([[1], [0]])) + np.kron(c[1], np.array([[0], [1]]))
B = A
solve_with_pic(A, B)

