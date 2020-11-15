from numba import cuda

@cuda.jit(device=True)#('void(float32[:], float32)')
def harm_osc_gpu(y, dydt, t): 
    # Try random noise
    # m = k = 1
    x, p = y[0], y[1]
    dydt[0] =  p
    dydt[1] = -x
