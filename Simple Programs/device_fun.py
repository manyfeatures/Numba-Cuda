from numba import cuda
import numpy as np

# It can be invoked ONLY from kernel and it return some value
@cuda.jit(device=True) 
def a_device_function(a, b):
    return a + b

@cuda.jit
def kernel(a,b,out):
	i = cuda.grid(1)
	if i < out.shape[0]: # In order not to exceed the shapes?
		out[i] = a_device_function(a[i], b[i])


a = cuda.to_device(np.random.rand(156)) # for example 156
b = cuda.to_device(3*np.ones(156))
out = cuda.device_array_like(a)


threadsperblock = 32
# In order to have redundant threads for safety we write below the line
blockspergrid = (out.size + (threadsperblock - 1)) // threadsperblock 

print(f" blockspergrid: {blockspergrid}")
kernel[blockspergrid, threadsperblock](a, b, out)
print(out.copy_to_host().shape)
print(out.copy_to_host())
