import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import pycuda as cuda
import pycuda.autoinit
import numpy as np
from numba import jit, njit, prange
from time import time
from collections import deque

# BATCH_SIZE = 1024
BATCH_SIZE = 98304
# GRID_SIZE = 16384
GRID_SIZE = 512
# GRID_SIZE = 24576
# GRID_SIZE = 49152
# GRID_SIZE = 1

modulo = ElementwiseKernel("unsigned int *x, unsigned int *y, int z", "x[i] = y[(i % z)] % x[i]", "modulo")
mod_full = SourceModule("""
__global__ void mod_full(unsigned int *x, unsigned int *y, unsigned int *z, unsigned int *stop, unsigned int width, unsigned int rows, unsigned int size) {
  const int n = blockIdx.x * 1024 + threadIdx.x;
  // const int s = n * width + blockIdx.y;
  if (n < width) {
    for (int i = blockIdx.y * width + n; i / width < rows && i / width < stop[n]; i += size * width) {
      // z[s] += (y[n] % x[i]) == 0;
      x[i] = (y[n] % x[i]) == 0;
    }
  }
}
""").get_function("mod_full")


limit = 16 # limit = int(input('Limit: '))
start_time = time()

@njit(parallel=True)
def fill_rows(start, stop, width):
    to_return = np.empty((stop - start, width), dtype=np.int32)
    for n in prange(start, stop):
        to_return[n - start] = n
    return to_return

most = np.uint32(0)
antiprimes = []

# @jit(parallel=True, forceobj=True)
def test_factor_batch(batch, mem):
    sqrts = np.sqrt(batch)
    ceils = np.ceil(sqrts).astype(dtype=np.uint32)
    test_mat = fill_rows(1, np.uint32(np.ceil(sqrts[-1])), batch.size)
    mat_gpu = gpuarray.to_gpu(test_mat, allocator=mem.allocate)
    res_mat = np.zeros((min(GRID_SIZE, test_mat.shape[0]), batch.size), dtype=np.uint16)
    # res_mat = np.zeros_like(test_mat, dtype=np.bool_)
    res_gpu = gpuarray.to_gpu(res_mat, allocator=mem.allocate)
    # modulo(mat_gpu, gpuarray.to_gpu(batch, mem.allocate), batch.size)
    mod_full(mat_gpu, gpuarray.to_gpu(batch, mem.allocate), res_gpu, gpuarray.to_gpu(ceils, allocator=mem.allocate),
             np.uint32(batch.size), np.uint32(test_mat.shape[0]), np.uint32(GRID_SIZE),
             block=(1024, 1, 1), grid=(BATCH_SIZE // 1024 + 1, GRID_SIZE))
    print(batch)
    print(test_mat)
    print(mat_gpu.get())
    res_mat = res_gpu.get()
    print(res_mat)
    # del mat_gpu
    num_factors = np.zeros(batch.size, dtype=np.uint32)
    for n in prange(0, batch.size):
        if batch[n] != 1:
            # count_facs = res_mat[:ceils[n] - 1, n]
            num_factors[n] = np.sum(res_mat[:, n])
            # num_factors[n] = np.sum(count_facs)
            num_factors[n] *= 2
    num_factors += np.equal(sqrts, ceils)
    return num_factors

def run_factor_batch(batch, mem):
    global most
    num_factors = test_factor_batch(batch, mem)
    for i in range(batch.size):
        if num_factors[i] > most:
            antiprimes.append(batch[i])
            most = num_factors[i]
            print(batch[i])


gpu_mem = cuda.tools.DeviceMemoryPool()
for n in range(1, limit + 1, BATCH_SIZE):
    n_max = limit + 1 if limit < n + BATCH_SIZE else n + BATCH_SIZE
    run_factor_batch(np.arange(n, n_max, dtype=np.uint32), gpu_mem)

print(antiprimes)
print(time() - start_time)
