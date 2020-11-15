# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux
from numba import cuda
from numba import *
from GPU_functions import harm_osc_gpu as func


@cuda.jit(device=True) # ('void(float32[:], float32[:], float32, int16)')
def comp_k1_term(init_conds, dydt, step, dt, k1):
    t_ = step*dt
    func(init_conds, dydt, t_)  # -30 ?
    for i in range(len(init_conds)):
          k1[i] = dydt[i] # dt*dydt or better multiply by dt in final equation?


@cuda.jit(device=True) # ('void(float32[:], float32[:], float32, int16)')
def comp_k2_term(init_conds, dydt, step, dt, k1, k2, buffer): 
    for i in range(len(init_conds)):
        buffer[i] = init_conds[i] + k1[i]/2.
    t_ =  step*dt + dt/2
    func(buffer, dydt, t_)  
    for i in range(len(init_conds)):
        k2[i] = dydt[i] # dt*dydt or better multiply by dt in final equation?


@cuda.jit(device=True) # ('void(float32[:], float32[:], float32, int16)')
def comp_k3_term(init_conds, dydt, step, dt, k2, k3, buffer):
    for i in range(len(init_conds)): 
        buffer[i] = init_conds[i] + k2[i]/2. 
    t_ =  step*dt + dt/2 
    func(buffer, dydt, dt)
    for i in range(len(init_conds)):
          k3[i] = dydt[i] # dt*dydt or better multiply by dt in final equation?

@cuda.jit(device=True) # ('void(float32[:], float32[:], float32, int16)')
def comp_k4_term(init_conds, dydt, step, dt, k3, k4, buffer):
    for i in range(len(init_conds)): 
        buffer[i] = init_conds[i] + k3[i] 
    t_ =  step*dt + dt/2 
    func(init_conds, dydt, t_)  
    for i in range(len(init_conds)):
          k4[i] = dydt[i] # dt*dydt or better multiply by dt in final equation?


@cuda.jit # ('void(float32[:], float32[:], float32, int16)')
def rk4_sol_cuda(init_conds, dydt, sol, dt, steps, k1, k2, k3, k4, buffer): # for intermediate calc-ns 
    """Simple Euler integrator"""    
    start = cuda.grid(1) # Each thread calculates one trajectory
    stride = cuda.gridsize(1) 

    # DEBUG FIRST THREAD
    #if start == 0:
    #   import pdb; pdb.set_trace()

    # Threads loop
    for i in range(start, init_conds.shape[0], stride): 
        ### DEBUG FIRST THREAD
        #if i == 0:
        #    import pdb; pdb.set_trace()
        ###

        # Time Loop
        #cuda.syncthreads() # Is it in the right place?  
        for step in range(1, steps):   # so ?         
            # shape: N, NE, dims  
            comp_k1_term(init_conds[i,:], dydt[i,:], step, dt, k1[i,:])
            #cuda.syncthreads() # Is it in the right place?
            comp_k2_term(init_conds[i,:], dydt[i,:], step, dt, k1[i,:], k2[i,:], buffer[i,:])
            #cuda.syncthreads() # Is it in the right place? 
            comp_k3_term(init_conds[i,:], dydt[i,:], step, dt, k2[i,:], k3[i,:], buffer[i,:])
            #cuda.syncthreads() # Is it in the right place? 
            comp_k4_term(init_conds[i,:], dydt[i,:], step, dt, k3[i,:], k4[i,:], buffer[i,:])
            #cuda.syncthreads() # Is it in the right place?             
            for j in range(init_conds.shape[1]): 
                    sol[i, j] =  init_conds[i, j]\
                                         + dt*(k1[i, j]\
                                               + 2*k2[i, j]\
                                               + 2*k3[i, j]\
                                               + k4[i, j])/6
                    # cuda.atomic.add???  
                    init_conds[i, j] = sol[i, j] 
                    cuda.syncthreads() # Is it in the right place?                    
            #cuda.syncthreads() # Is it in the right place?                    
        cuda.syncthreads() # Is it in the right place?