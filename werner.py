import numpy as np
import copy
import scipy
import scipy.stats
from scipy import tensordot


def diag_axis(self, axis=-1):
    """ Diagonalize axis.
    Similar to using :code:`np.diag(a)`, however for higher dimensional
    arrays.
    Doing this on a dense array would be prohibitevly expensive as
    a lot of zeros would be introduced. However for a sparse array it
    simply means replicating the appropriate row in the :code:`coords`
    array.
    Parameters
    ----------
    axis : integer
        The axis to be diagonalized.
    Returns
    -------
    COO
        The output of the computation. A view of all unchanged
        attributes is returned.
    See also
    --------
    np.diag : NumPy equivalent for diagonalizing 1D arrays
    """
    out = copy.copy(self)
    # normalize axis index
    naxis = list(range(out.ndim))[axis]
    shape = list(out.shape)
    shape.insert(naxis, shape[naxis])

    out.coords = np.insert(out.coords, naxis, out.coords[naxis, :], 0)
    out.shape = tuple(shape)
    return out



def tensordot2(A, B, sum=None, multiply=None):
    """ Tensordot that supports elementwise multiplication of axes
    without a subsequent sum-contraction.
    A sum contraction "can be prevented" if the corresponding axis is
    diagonalized. This principle can be seen in the most simple case by
    comparing
    ..code::
        a = np.arange(5)
        b = np.arange(5)
        a @ b           # sum-contracted to scalar
        np.diag(a) @ b  # vector with elementwise products
    Diagonalizing axes on a dense array would of course be prohibitevly
    costly but it is really cheap for sparse matrices.
    Parameters
    ----------
    A, B : COO
        The input arrays.
    sum : list[list[int]]
        The axes to multiply and sum-contract over.
    multiply : list[list[int]]
        The axes to multiply over.
    Returns
    -------
    COO
        The output of the computation.
    See Also
    --------
    einsum : Einstein summation function using this
    COO.diag_axis : Diagonalize axes of sparse array
    """
    if sum is None:
        sum = [[], []]
    else:
        sum = list(sum)

    if multiply is None:
        multiply = [[], []]
    else:
        multiply = list(multiply)

    # For each multiply[0] we are adding one axis, thus we need to increment
    # all following items by one: (0, 1, 2) -> (0, 2, 4)
    # We need to account that the array may be unsorted
    idx = np.argsort(multiply[0])
    post_multiply = multiply[0]
    for i, v in enumerate(idx):
        post_multiply[v] += i

    for i in post_multiply:
        A = A.diag_axis(i)

    sum[0] += post_multiply
    sum[1] += multiply[1]

    return tensordot(A, B, axes=sum)


def einsum(ops, *args):
    """ Evaluates the Einstein summation convention on the operands.
    Parameters
    ----------
    ops : string
        Specifies the subscripts for summation.
    *args : COO
        These are the arrays for the operation. Only exactly two are supported.
    Returns
    -------
    COO
        The output of the computation.
    See Also
    --------
    numpy.einsum : NumPy equivalent function
    tensordot2 : Non-contracting tensordot implementation
    """

    if len(args) != 2:
        raise ValueError("Currently only two operands are supported")

    inops, outops = ops.split('->')
    inops = inops.split(',')

    # All indices that are in input AND in output are multiplies
    multiplies = sorted(list(set(inops[0]) & set(inops[1]) & set(outops)))
    # All indices that are in input BUT NOT in output are sum contractions
    sums = sorted(list((set(inops[0]) & set(inops[1])) - set(outops)))

    # Map sums and indices to axis integers
    multiplies = [[inop.find(x) for x in multiplies] for inop in inops]
    sums = [[inop.find(x) for x in sums] for inop in inops]

    # Find output axes in input axes for final transpose
    # Values very likely lie outside of output tensor shape, so
    # just map them values to their rank (index in ordered list)
    transpose = [''.join(inops).find(x) for x in outops]
    transpose = scipy.stats.rankdata(transpose).astype(int) - 1

    return tensordot2(*args, sum=sums, multiply=multiplies).transpose(transpose)
