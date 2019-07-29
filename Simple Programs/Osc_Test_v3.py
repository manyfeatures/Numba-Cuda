#SET NUMBA_ENABLE_CUDASIM=1
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer


blocks_per_grid = 128
threads_per_block = 1024

@cuda.jit(device=True)#('void(float32[:], float32)')
def harm_osc_gpu(y):
    x,p = y[0], y[1]
    dydt = (p,-x)
    return dydt


@cuda.jit#('void(float32[:], float32[:], float32, int16)')
def euler_sol_cuda(y0, sol, dt, steps): 
    start = cuda.grid(1)      
    stride = cuda.gridsize(1) 

    # Threads loop
    for i in range(start, y0.shape[0], stride): # stride = threads_per_block * blocks_per_grid
        
        ### DEBUG FIRST THREAD
        #if start == 10 and i == 10:
        #    from pdb import set_trace; set_trace()
        ###

        # Time Loop 
        for k in range(steps):
            dydt = harm_osc_gpu(y0[i]) #
            # Loop for assignment of x and p
            for j in range(y0.shape[1]): # or do it without loop
                sol[i, j] =  y0[i, j] + dt*dydt[j]
                y0[i,j] = sol[i, j] 

# CPU
n = 128*1024 # number of trajectories
dt = np.float32(0.1)
steps = np.int16(1000)
#y0 = np.array([10,0], dtype="float64")
#y0 = y0.repeat(n).reshape(2,n).transpose() # 1000 x 2
y0 = np.random.rand(n,2)
#y0[0,:] = [1,2] # Change one
sol = np.zeros((n,y0.shape[1]),  dtype="float64")

# To GPU
#dt_d = cuda.to_device(dt)
#steps_d = cuda.to_device(steps)
y0_d = cuda.to_device(y0)
sol_d = cuda.device_array_like(sol)

#print(f"{y0}")
start = timer()
euler_sol_cuda[blocks_per_grid, threads_per_block](y0_d, sol_d, dt, steps)
print(sol_d.copy_to_host())
t_ = timer() - start
print ("Time consumed %f s" % t_)
