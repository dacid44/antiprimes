import pycuda.gpuarray as gpuarray
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
from pycuda.elementwise import ElementwiseKernel

matrix_size = (5,)
a = np.random.randint(2, size=matrix_size, dtype=np.int32)
b = np.random.randint(2, size=matrix_size, dtype=np.int32)

print(a)
print(b)

a_gpu = gpuarray.to_gpu(a)
b_gpu = gpuarray.to_gpu(b)

eq_checker = ElementwiseKernel(
        "int *x, int *y, int *z",
        "z[i] = x[i] == y[i]",
        "equality_checker")

c_gpu = gpuarray.empty_like(a_gpu)
eq_checker(a_gpu, b_gpu, c_gpu)

print(c_gpu)