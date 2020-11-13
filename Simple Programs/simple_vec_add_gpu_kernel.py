# SET NUMBA_ENABLE_CUDASIM=1 for debugging
# export NUMBA_ENABLE_CUDASIM=1 in linux


from numba import cuda 
import math
import matplotlib.pyplot as plt
import numpy as np

@cuda.jit # ('void(float32[:], float32[:], float32, int16)')
def gpu_sum(d_a, d_b, d_out): 
    """Simple Euler integrator"""    
    start = cuda.grid(1) # Each thread calculates one trajectory
    stride = cuda.gridsize(1) 

    # Threads loop
    up_lim = len(d_a)
    for i in range(start, up_lim, stride):
        d_out[i] = d_a[i] + d_b[i]
        

a = 10*np.random.standard_normal(256) # float64 by default
b = np.arange(0,256, dtype=np.float64)

# Copy data to device
d_a = cuda.to_device(a) 
d_b = cuda.to_device(b) 
d_out = cuda.device_array_like(a)
#d_c = cuda.device_array((N, NE, 6), dtype=np.float64)

# Call kernel
gpu_sum[16,16](d_a, d_b, d_out)

# Copy output to host
out = d_out.copy_to_host()
print(out)
plt.imshow(out.reshape(16,16)); plt.show()
