# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.cm as cm

blocks_per_grid = 1
threads_per_block = 128

@cuda.jit(device=True)#('void(float32[:], float32)')
def harm_osc_gpu(y): 
    # Try random noise
    # m = k = 1
    x, p = y[0], y[1]
    dydt = (p,-x)
    return dydt


@cuda.jit#('void(float32[:], float32[:], float32, int16)')
def euler_sol_cuda(y0, sol, dt, steps): 
    start = cuda.grid(1)      
    stride = cuda.gridsize(1) 

    # Threads loop
    for i in range(start, y0.shape[0], stride): # stride = threads_per_block * blocks_per_grid        
        
        ### DEBUG THREAD
        #if start == 10 and i == 10:
        #    from pdb import set_trace; set_trace()

        # Time Loop 
        for k in range(steps):
            dydt = harm_osc_gpu(y0[i]) #
            # Loop for assignment of x and p
            for j in range(y0.shape[1]): # or do it without loop
                sol[i, j] =  y0[i, j] + dt*dydt[j]
                y0[i,j] = sol[i, j] 


def show_plots(init, sol, single_sol = []):
    # Plot different color for each oscillator
    colors = cm.rainbow(np.linspace(0, 1, len(init)))

    # plt.subplot(121)
    # if single_sol.size != 0:
    #     plt.scatter(single_sol[:,0], single_sol[:,1])

    # plt.subplot(122)
    # if single_sol.size != 0:
    #     plt.scatter(single_sol[:,0], single_sol[:,1])

    for y0, y_end, c in zip(init, sol, colors):
        plt.subplot(121)
        plt.scatter(y0[0], y0[1], color=c)
        plt.subplot(122)
        plt.scatter(y_end[0], y_end[1], color=c)

# On CPU
n = blocks_per_grid * threads_per_block  # number of trajectories
dt = np.float32(0.001)
STEPS = np.int16(3000)
# Init conditions
y0 = np.random.rand(n,2) 
#sol_h = np.zeros((n,y0.shape[1]),  dtype="float64")

# To GPU
y0_d = cuda.to_device(y0)
sol_d = cuda.device_array_like(y0)

start = timer()
euler_sol_cuda[blocks_per_grid, threads_per_block](y0_d, sol_d, dt, STEPS)
sol_h = sol_d.copy_to_host()
t_ = timer() - start
print ("Time consumed %f s" % t_)

#CPU sol
# print("CPU")
# TRAJ_NUM = np.random.randint(y0.shape[0], size=1)[0]
# print(TRAJ_NUM)
# single_sol = test_trajectory(harm_osc, y0[TRAJ_NUM, :], dt)
#
# def test_trajectory(f, y0, dt):
#     # verify single trajectory
#     from scipy.integrate import odeint
#     t = np.linspace(0, STEPS*dt, STEPS)
#     sol = odeint(f, y0, t)
#     return sol
#
# def harm_osc(y, t): 
#     # m = k = 1
#     x, p = y[0], y[1]
#     dydt = (p,-x)
#     return dydt

# Plot
show_plots(y0, sol_h) # single_sol can be added
plt.show()

