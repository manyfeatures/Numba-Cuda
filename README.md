# Cuda

## Numba implementation

## 1050 TI Specification
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
Total amount of constant memory: 65536 bytes

Total amount of shared memory per block: 49152 bytes

Total number of registers available per block: 65536

Warp size: 32

Maximum number of threads per multiprocessor: 2048

Maximum number of threads per block: 1024

Max dimension size of a thread block (x,y,z): (1024, 1024, 64)

Max dimension size of a grid size (x,y,z): (2147483647, 65535, 65535)

Maximum memory pitch: 2147483647 bytes

Texture alignment: 512 bytes

Concurrent copy and kernel execution: Yes with 2 copy engine(s)

Run time limit on kernels: No

Integrated GPU sharing Host Memory: No

Support host page-locked memory mapping: Yes

Alignment requirement for Surfaces: Yes

Device has ECC support: Disabled

CUDA Device Driver Mode (TCC or WDDM): WDDM (Windows Display Driver Model)

Device supports Unified Addressing (UVA): Yes

Device PCI Domain ID / Bus ID / location ID: 0 / 1 / 0

Compute Mode:

< Default (multiple host threads can use ::cudaSetDevice() with device simultaneously) >

deviceQuery, CUDA Driver = CUDART, CUDA Driver Version = 9.1, CUDA Runtime Version = 8.0, NumDevs = 1, Device0 = GeForce GTX 1050 Ti

Result = PASS 
