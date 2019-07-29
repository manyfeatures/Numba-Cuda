import numpy as np
from numba import cuda
from numba import *
from timeit import default_timer as timer
import numpy as np
import scipy 

# GPU params?
threads_per_block = 16
blocks_per_grid = 32

# CPU Params?
f = 4000 # a.u.
w = 2*np.pi*f #
c = 137 
k = np.array([w/c, 0, 0]) # pulse moves along x-axis 
A0 = 3
m = 1  
e = -1  
tau = 30 
Ze = 1
r_targ = [0, 0, 0]

eps = 1e-1 # choose more reasonable value


# Auxiliary function
# Move them to different file
def vec_pot(r,t): # r
    A = [0, A0*np.cos(np.dot(k,r) - w*t) * np.exp(-(np.pi*t)**2/tau**2), 0]
    return A

def grad_vec_pot_t(r,t): # r is numpy array
    dA = np.array([0, A0 * w * np.sin(np.dot(k,r)-w*t) * np.exp(-(np.pi*t)**2/tau**2), 0])
    return dA

def force_col(r):
    r_diff = np.array([r[0]-r_targ[0], r[1]-r_targ[1], r[2]-r_targ[2]])
    F = e*Ze*r_diff/(eps+(np.linalg.norm(r_diff))**3)
    return F
                     
def magnet_field(r,t):
    B = [0, 0, -k[0]*A0*np.sin(np.dot(k,r)-w*t) * np.exp(-(np.pi*t)**2/tau**2)]
    return np.array(B)

def magnet_force(r, v, t):
    B = magnet_field(r, t)
    F = e*np.array([B[2]*v[1], -B[2]*v[0], 0])/c
    return F

def velocity(p): # arg should be numpy array!
    return p*c/np.sqrt(
        (m*c)**2 +(np.linalg.norm(p))**2
        )

def hamil_eq(y, t):
    x0,x1,x2,p0,p1,p2 = y[0], y[1], y[2], y[3], y[4], y[5] # can be removed
    r = np.array([x0, x1, x2])
    p = np.array([p0, p1, p2])
    v = velocity(p)
    F_col = force_col(r)
    dAdt = grad_vec_pot_t(r,t)
    F_B = magnet_force(r, v, t)
    dydt = (
                v[0], # ok 
                v[1],
                v[2],
                F_col[0] - e*dAdt[0]/c + F_B[0],
                F_col[1] - e*dAdt[1]/c + F_B[1],
                F_col[2] - e*dAdt[2]/c + F_B[2]
           )

    return np.array(dydt)

# GPU functions    
#vec_pot_gpu = cuda.jit(device=True)(vec_pot)
#grad_vec_pot_t_gpu = cuda.jit(device=True)(grad_vec_pot_t)
#force_col_gpu = cuda.jit(device=True)(force_col)
#magnet_field_gpu = cuda.jit(device=True)(magnet_field)
#magnet_force_gpu = cuda.jit(device=True)(magnet_force)
#velocity_gpu = cuda.jit(device=True)(velocity)
hamil_eq_gpu = cuda.jit(device=True)(hamil_eq)


@cuda.jit('void(float32[:], float32[:], float32, int16)')
def euler_sol_cuda(y0, sol, dt, steps): # all input arrays are of numpy 
    start = cuda.grid(1)      # 1 = one dimensional thread grid, returns a single value
    stride = cuda.gridsize(1) # threads_per_block * blocks_per_grid --- number of threads

    ### DEBUG FIRST THREAD
    #if start == 0:
    #   from pdb import set_trace; set_trace()
    ###

        # Threads loop
    for i in range(start, y0.shape[0], stride): # stride = threads_per_block * blocks_per_grid
        # Time Loop 
        for k in range(steps):
            dydt = hamil_eq_gpu(y0[i], 0) #
            # Loop for assignment of x and p
            for j in range(y0.shape[1]): # or do it without loop
                #dt*dydt[j] 
                sol[i, j] =  y0[i, j] + dt*dydt[j]
                #cuda.atomic.add(sol, (i,j), dt*dydt[j] ) # sol[i, j] =  y0[i, j] + dt*dydt[j]  
                y0[i,j] = sol[i, j] 


dt = np.float32(0.1)
steps = np.int16(1000)
n = 100 # number of trajectories
y0 = np.array([0,0,0,0,10,0], dtype="float32")
y0 = y0.repeat(n).reshape(6,n).transpose() # 1000 x 6
#y0[0,:] = [1,2] # Change one
sol = np.zeros((n,y0.shape[1]),  dtype="float32")
print("sol:", sol.shape)
# To GPU
y0_device = cuda.to_device(y0)
sol_device = cuda.device_array_like(sol)

start = timer()
euler_sol_cuda[blocks_per_grid, threads_per_block](y0_device, sol_device, dt, steps)
print(sol_device.copy_to_host())
t_ = timer() - start
print ("Time consumed %f s" % t_)