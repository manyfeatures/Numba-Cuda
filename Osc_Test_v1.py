import numpy as np
from numba import cuda
from numba import *

threads_per_block = 128
blocks_per_grid = 32

k = 1
m = 1

def harm_osc(y,t):
#def harm_osc(t, y):
    x = y[0]
    p = y[1]
    dydt = [p/m,-k*x]
    return dydt

harm_osc_gpu = cuda.jit(device=True)(harm_osc)

@cuda.jit
def euler_sol_cuda(y0, sol): # all input arrays are of numpy 

    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # threads_per_block * blocks_per_grid --- number of threads

    # assuming x and y inputs are same length
    for i in range(start, y0.shape[0], stride): # stride = threads_per_block * blocks_per_grid
        dydt = harm_osc_gpu((y0[i,0],y0[i,1]), 0)
        for j in range(2):
            #print(type(dydt))
            sol[i, j] = int(dydt[j])

n = 1000 # number of trajectories
y0 = np.array([10,0], dtype="float32")
y0 = y0.repeat(n).reshape(2,n).transpose() # 1000 x 2
sol = np.zeros((n,y0.shape[1]),  dtype="float32")

y0_device = cuda.to_device(y0)
sol_device = cuda.device_array_like(sol)

euler_sol_cuda[blocks_per_grid, threads_per_block](y0_device, sol_device)
print(sol_device.copy_to_host())
