import numpy as np
from numba import cuda



@cuda.jit
def vec_add(A, B, out):
    x = cuda.threadIdx.x
    bx = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    if x == 1 and bx == 3:
        from pdb import set_trace; set_trace()
    i = bx * bdx + x
    out[i] = A[i] + B[i]

A = np.random.rand(4096)
B = np.random.rand(4096)
C = np.zeros(4096)
vec_add[64, 64](A, B, C)