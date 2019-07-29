#SET NUMBA_ENABLE_CUDASIM=1
import numpy as np
from numba import cuda
from numba import *


@cuda.jit(device=True)#('void(float32[:], float32)')
def comp_array_2(arr): # y is a 3d vector
    """Returns y + 1"""
    arr = arr+1
    return arr

@cuda.jit(device=True)#('void(float32[:], float32)')
def comp_array_1(arr_1, arr_2): # y is a 3d vector
    """Returns y*(y + 1)"""
    arr_1[0] = 1
    arr_1[1] = 3
    arr_1[2] = 1
    #return arr_2
    pass       

@cuda.jit
def euler_sol_cuda(init, sol, arr_1, arr_2): 
    start = cuda.grid(1)      
    stride = cuda.gridsize(1) 

    # Threads loop
    for i in range(start, init.shape[0], stride): 
        # sol[i,:] = comp_array_1(arr_1[i,:], arr_2[i,:]) #
        comp_array_1(arr_1[i,:], arr_2[i,:])
        #sol[i,0] = 1

print(__name__)

if __name__=="__main__":
    # CPU
    n = 5 
    blocks_per_grid = 32
    threads_per_block = 64

    # To GPU
    d_init = cuda.to_device(np.random.rand(n,3))
    d_sol = cuda.device_array_like(d_init)
    d_arr_1 = cuda.device_array_like(d_init)
    d_arr_2 = cuda.device_array_like(d_init)

    euler_sol_cuda[blocks_per_grid, threads_per_block](d_init, d_sol, d_arr_1, d_arr_2)
    print(d_sol.copy_to_host())
    print(d_arr_1.copy_to_host())
