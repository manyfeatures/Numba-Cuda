# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux
from numba import cuda
from numba import *
from gpu_functions import harm_osc_gpu as func


# Euler integrator
@cuda.jit # ('void(float32[:], float32[:], float32, int16)')
def euler_sol_cuda(y0, sol, dydt, dt, steps): 
    start = cuda.grid(1)      
    stride = cuda.gridsize(1) 

    # Threads loop
    for i in range(start, y0.shape[0], stride): # stride = threads_per_block * blocks_per_grid        
        # Time Loop 
        cuda.syncthreads() # Is it in the right place? 
        for step in range(1, steps):
            # derivative function
            func(y0[i], dydt[i], steps*dt) # for harmnic ocsillator time is unimportant here
            # Loop for assignment of x and p
            for j in range(y0.shape[1]): # or do it without loop
                sol[i, j] =  y0[i, j] + dt*dydt[i, j] 
                y0[i,j] = sol[i, j]  # upd init conds
            cuda.syncthreads() # Is it in the right place? 
    cuda.syncthreads() # Is it in the right place? 