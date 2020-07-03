# Cuda

### Useful Links

 - https://cuda-tutorial.readthedocs.io/en/latest/tutorials/tutorial01/


## Numba implementation

## 1050 TI Specification
```
`From: https://devtalk.nvidia.com/default/topic/1029700/cuda-for-geforce-gtx-1050-ti/`

- My laptop has a 1050 Ti and this is the deviceQuery output:
- Detected 1 CUDA Capable device(s)

- Device 0: "GeForce GTX 1050 Ti"
- CUDA Driver Version / Runtime Version 9.1 / 8.0
- CUDA Capability Major/Minor version number: 6.1
- Total amount of global memory: 4096 MBytes (4294967296 bytes)
- ( 6) Multiprocessors, (128) CUDA Cores/MP: 768 CUDA Cores
- GPU Max Clock rate: 1620 MHz (1.62 GHz)
- Memory Clock rate: 3504 Mhz
- Memory Bus Width: 128-bit
- L2 Cache Size: 1048576 bytes
- Maximum Texture Dimension Size (x,y,z) 1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
- Maximum Layered 1D Texture Size, (num) layers 1D=(32768), 2048 layers
- Maximum Layered 2D Texture Size, (num) layers 2D=(32768, 32768), 2048 layers
- Total amount of constant memory: 65536 bytes
- Total amount of shared memory per block: 49152 bytes
- Total number of registers available per block: 65536
- Warp size: 32
- Maximum number of threads per multiprocessor: 2048
- Maximum number of threads per block: 1024
- Max dimension size of a thread block (x,y,z): (1024, 1024, 64)
- Max dimension size of a grid size (x,y,z): (2147483647, 65535, 65535)
- Maximum memory pitch: 2147483647 bytes
- Texture alignment: 512 bytes
- Concurrent copy and kernel execution: Yes with 2 copy engine(s)
- Run time limit on kernels: No
- Integrated GPU sharing Host Memory: No
- Support host page-locked memory mapping: Yes
- Alignment requirement for Surfaces: Yes
- Device has ECC support: Disabled
- CUDA Device Driver Mode (TCC or WDDM): WDDM (Windows Display Driver Model)
- Device supports Unified Addressing (UVA): Yes
- Device PCI Domain ID / Bus ID / location ID: 0 / 1 / 0
- Compute Mode:
< Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >
deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.1, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = GeForce GTX 1050 Ti
Result = PASS 
```

### CUDA code workflow
```
 Following is the common workflow of CUDA programs.

    Allocate host memory and initialized host data
    Allocate device memory
    Transfer input data from host to device memory
    Execute kernels
    Transfer output from device memory to host

```

### CUDA functions

```
cudaMalloc(void **devPtr, size_t count);
//cudaMalloc() allocates memory of size count in the device memory and updates the device pointer devPtr to the //allocated memory.
```




### Compile CUDA Program
```
#include <stdio.h>


__global__ void cuda_hello(){
        printf("Hello World from GPU!\n");
}

int main(){
        cuda_hello<<<1,1>>>();
        return 0;
}
```

Compile code above 
```
nvcc hello.cu -o hello
```

### Some code notes

The `__global__` specifier indicates a function that runs on device (GPU). Such function can be called through host code, e.g. the `main()` function in the example, and is also known as "kernels".


# C insertions

###### Pointers as arguments

`Pass By Value: void fcn(int foo)`

When passing by value, you get a copy of the value. If you change the value in your function, the caller still sees the original value regardless of your changes

`Pass By Pointer to Value: void fcn(int* foo)`

Passing by pointer gives you a copy of the pointer - it points to the same memory location as the original. This memory location is where the original is stored. This lets you change the pointed-to value. However, you can't change the actual pointer to the data since you only received a copy of the pointer.

`Pass Pointer to Pointer to Value: void fcn(int** foo)`

You get around the above by passing a pointer to a pointer to a value. As above, you can change the value so that the caller will see the change because it's the same memory location as the caller code is using. For the same reason, you can change the pointer to the value. This lets you do such things as allocate memory within the function and return it; `&arg2 = calloc(len);`. You still can't change the pointer to the pointer, since that's the thing you recieve a copy of.


###### Vector addition in C (Compare with CUDA)
```
#define N 10000000

void vector_add(float *out, float *a, float *b, int n) {
    for(int i = 0; i < n; i++){
        out[i] = a[i] + b[i];
    }
}

int main(){
    float *a, *b, *out; 

    // Allocates memory and returns a pointer to it.
    a   = (float*)malloc(sizeof(float) * N); // size in bytes
    b   = (float*)malloc(sizeof(float) * N);
    out = (float*)malloc(sizeof(float) * N);

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f; b[i] = 2.0f;
    }

    // Main function
    vector_add(out, a, b, N);
}
```