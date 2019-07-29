from numba import cuda
import numpy as np
from timeit import default_timer as timer

# How much memory does it use?
N = 128
blockspergrid = 4 # Memory !!! If  #blocks x #threads < matrix shape!!! Does not work. Why?
threadsperblock = 32

# when types are specified it seems doesn't operate properly
@cuda.jit#('void(float32[:,:], float32[:,:], float32[:,:])') # Assign types explicitly
def kernel_1(a,b,out):
    x = cuda.grid(1)
    if x <= out.shape[0]: #and y <= out.shape[1]: # In order not to exceed the shapes?
    	out[x] = a[x] + b[x]
        
a = cuda.to_device(np.random.rand(N))
b = cuda.to_device(3*np.ones((N)))
out = cuda.device_array_like(a) # Empty array

start = timer()
kernel_1[blockspergrid, threadsperblock](a, b, out)
t_ = timer() - start

print ("Time consumed %f s" % t_)
print(a.copy_to_host())
print(b.copy_to_host())
print(out.copy_to_host())
