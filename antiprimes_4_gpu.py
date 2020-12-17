import pycuda.gpuarray as gpuarray
from pycuda.compiler import SourceModule
import pycuda as cuda
import pycuda.autoinit
import numpy as np
from numba import njit, prange
from time import time

BATCH_MULTIPLIER = 64
BATCH_SIZE = int(196608 * BATCH_MULTIPLIER)
GRID_SIZE = int(256 // BATCH_MULTIPLIER)
GEN_TESTS = False

limit = 1000000000  # limit = int(input('Limit: '))

# noinspection PyStringFormat
mod_gen = SourceModule("""
__global__ void mod_gen({1}unsigned int *y, unsigned {0} *z, unsigned int *stop, int width) {{
  const int n = blockIdx.x * 1024 + threadIdx.x;
  const int s = blockIdx.y * width + n;
  const unsigned short size = gridDim.y;
  if (n < width) {{
    for (int i = blockIdx.y + 1; i < stop[n] - 1; i += size) {{
      z[s] += (y[n] % {2}) == 0;
    }}
  }}
}}
""".format('short' if limit <= 1000000000 else 'int',
           *(('', 'i') if GEN_TESTS else ('unsigned int *x, ', 'x[i]'))
           )).get_function("mod_gen")
n_data_type = np.uint16 if limit <= 1000000000 else np.uint32

start_time = time()

most = np.uint32(0)
antiprimes = []

@njit(parallel=True)
def count_factors(results):
    to_return = np.zeros(results.shape[1], dtype=np.uint32)
    for n in prange(results.shape[1]):
        to_return[n] = np.sum(results[:, n])
        to_return[n] *= 2
    return to_return

def test_factor_batch(batch, mem):
    sqrts = np.sqrt(batch)
    ceils = np.ceil(sqrts).astype(dtype=np.uint32)
    if GEN_TESTS:
        mod_args = (gpuarray.to_gpu(batch, mem.allocate),)
    else:
        test_mat = np.arange(1, ceils[-1], dtype=np.uint32)
        mat_gpu = gpuarray.to_gpu(test_mat, allocator=mem.allocate)
        mod_args = (mat_gpu, gpuarray.to_gpu(batch, mem.allocate))
    res_mat = np.zeros((min(GRID_SIZE, ceils[-1] - 1), batch.size), dtype=n_data_type)
    res_gpu = gpuarray.to_gpu(res_mat, allocator=mem.allocate)
    mod_gen(*mod_args, res_gpu, gpuarray.to_gpu(ceils, allocator=mem.allocate),
            np.uint32(batch.size), block=(1024, 1, 1), grid=(BATCH_SIZE // 1024 + 1, GRID_SIZE))
    res_mat = res_gpu.get()
    if not GEN_TESTS:
        del mat_gpu
    del res_gpu
    num_factors = count_factors(res_mat)
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
