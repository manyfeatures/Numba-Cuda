# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stuff import *
from Euler_template import *


blocks_per_grid = 4
threads_per_block = 32

def show_plots(init, sol, single_sol = []):
    # Plot different color for each oscillator
    colors = cm.rainbow(np.linspace(0, 1, len(init)))
    for y0, y_end, c in zip(init, sol, colors):
        plt.subplot(121)
        plt.scatter(y0[0], y0[1], color=c)
        plt.subplot(122)
        plt.scatter(y_end[0], y_end[1], color=c)


def main():
    # On CPU
    n = blocks_per_grid * threads_per_block  # number of trajectories
    dt = np.float32(0.001)
    STEPS = np.int16(300)
    # Init conditions
    y0 = np.random.rand(n,2) 

    # To GPU
    y0_d = cuda.to_device(y0)
    sol_d = cuda.device_array(y0.shape, dtype="float64")
    dydt_d = cuda.device_array(y0.shape, dtype="float64")
    print(sol_d.dtype)

    # Run on GPU
    start = timer()
    euler_sol_cuda[blocks_per_grid, threads_per_block](y0_d, dydt_d, sol_d, dt, STEPS)
    sol_h = sol_d.copy_to_host()
    t_ = timer() - start
    print ("Time consumed %f s" % t_)
    print(sol_h)

    # Plot
    show_plots(y0, sol_h, single_sol = []) # single_sol can be added
    plt.show()


if __name__ == "__main__":
    #import cProfile as cprofile
    #cprofile.run("main()", sort="time")
    #with cuda.profiling():
    main()