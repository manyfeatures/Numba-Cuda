# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux
import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from stuff import *
import gpu_integrators as integrators 

blocks_per_grid = 128
threads_per_block = 32
solver = "RK4" #"Euler"
# @cuda.jit(device=True)#('void(float32[:], float32)')
# def harm_osc_gpu(y): 
#     # Try random noise
#     # m = k = 1
#     x, p = y[0], y[1]
#     dydt = (p,-x)
#     return dydt

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


def main():
    # On CPU
    n = blocks_per_grid * threads_per_block  # number of trajectories
    dt = np.float32(0.001)
    STEPS = np.int16(300)
    # Init conditions
    y0 = np.random.rand(n,2) 
    #sol_h = np.zeros((n,y0.shape[1]),  dtype="float64")

    # To GPU
    y0_d = cuda.to_device(y0)
    sol_d = cuda.device_array(y0.shape, dtype="float64")
    dydt_d = cuda.device_array(y0.shape, dtype="float64")
    if solver == "RK4":
        ########### rk4 arrays ################
        d_k1 = cuda.device_array_like(y0)
        d_k2 = cuda.device_array_like(y0)
        d_k3 = cuda.device_array_like(y0)
        d_k4 = cuda.device_array_like(y0)

        d_buffer = cuda.device_array_like(y0)
        ########### rk4 arrays ################
    print(sol_d.dtype)

    # Run on GPU
    start = timer()
    if solver == "Euler":
        integrators.euler_sol_cuda[blocks_per_grid, threads_per_block](y0_d, dydt_d, sol_d, dt, STEPS)
    elif solver == "RK4":
        integrators.rk4_sol_cuda[blocks_per_grid, threads_per_block](y0_d, dydt_d, sol_d, dt, STEPS, d_k1, d_k2, d_k3, d_k4, d_buffer)
    sol_h = sol_d.copy_to_host()
    t_ = timer() - start
    print ("Time consumed %f s" % t_)

    print(sol_h)

    #CPU sol
    # print("CPU")
    # TRAJ_NUM = np.random.randint(y0.shape[0], size=1)[0]
    # print(TRAJ_NUM)
    # single_sol = test_trajectory(harm_osc, y0[TRAJ_NUM, :], dt, STEPS)
    # print(type(single_sol))

    # Plot
    show_plots(y0, sol_h, single_sol = []) # single_sol can be added
    plt.show()


# def test_trajectory(f, y0, dt, STEPS):
#     # verify single trajectory
#     from scipy.integrate import odeint
#     t = np.linspace(0, STEPS*dt, STEPS)
#     sol = odeint(f, y0, t)
#     return sol

# def harm_osc(y, t): 
#     # m = k = 1
#     x, p = y[0], y[1]
#     dydt = (p,-x)
#     return dydt

# folder = './distrs'
# filename = f'GPU_trajectory_A_{str(A0)}_d_{str(d)}_w_{str(w)}.npy'
# path = os.path.join(folder, filename) 
# if os.path.exists(folder): 
#     np.save(path, h_sol_all) # Without init conds. Do it more elegant
# else: # create folder
#     os.makedirs(folder)
#     np.save(path, h_sol_all) # Without init conds. Do it more elegant


if __name__ == "__main__":
    #import cProfile as cprofile
    #cprofile.run("main()", sort="time")
    #with cuda.profiling():
    main()