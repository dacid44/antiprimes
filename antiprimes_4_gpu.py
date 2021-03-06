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
PRINT_FILE = True

limit = 1000000000000  # limit = int(input('Limit: '))

# noinspection PyStringFormat
mod_gen = SourceModule("""
__global__ void mod_gen({1}unsigned {{0}} *y, unsigned {0} *z, unsigned {{0}} *stop, int width) {{{{
  const int n = blockIdx.x * 1024 + threadIdx.x;
  const int s = blockIdx.y * width + n;
  const unsigned short size = gridDim.y;
  if (n < width) {{{{
    for (int i = blockIdx.y{2}; i < stop[n]{3}; i += size) {{{{
      z[s] += (y[n] % {4}) == 0;
    }}}}
  }}}}
}}}}
""".format('short' if limit <= 1000000000 else 'int',
           *(('', ' + 1', '', 'i') if GEN_TESTS else ('unsigned {0} *x, ', '', ' - 1', 'x[i]'))
           ).format('int' if limit <= 1000000000 else 'long long')).get_function("mod_gen")
n_data_type = np.uint16 if limit <= 1000000000 else np.uint32
m_data_type = np.uint32 if limit <= 1000000000 else np.uint64

start_time = time()

most = m_data_type(0)
antiprimes = []

@njit(parallel=True)
def count_factors(results):
    to_return = np.zeros(results.shape[1], dtype=m_data_type)
    for n in prange(results.shape[1]):
        to_return[n] = np.sum(results[:, n])
        to_return[n] *= 2
    return to_return

def test_factor_batch(batch, mem):
    sqrts = np.sqrt(batch)
    ceils = np.ceil(sqrts).astype(dtype=m_data_type)
    if GEN_TESTS:
        mod_args = (gpuarray.to_gpu(batch, mem.allocate),)
    else:
        test_mat = np.arange(1, ceils[-1], dtype=m_data_type)
        mat_gpu = gpuarray.to_gpu(test_mat, allocator=mem.allocate)
        mod_args = (mat_gpu, gpuarray.to_gpu(batch, mem.allocate))
    res_mat = np.zeros((min(GRID_SIZE, ceils[-1] - 1), batch.size), dtype=n_data_type)
    res_gpu = gpuarray.to_gpu(res_mat, allocator=mem.allocate)
    mod_gen(*mod_args, res_gpu, gpuarray.to_gpu(ceils, allocator=mem.allocate),
            m_data_type(batch.size), block=(1024, 1, 1), grid=(BATCH_SIZE // 1024 + 1, GRID_SIZE))
    res_mat = res_gpu.get()
    if not GEN_TESTS:
        del mat_gpu
    del res_gpu
    num_factors = count_factors(res_mat)
    num_factors += np.equal(sqrts, ceils)
    return num_factors

@njit(parallel=True)
def is_antiprime(t_most, candidates, num_factors):
    is_possible = num_factors > t_most
    to_return = []
    for num, facs in zip(candidates[is_possible], num_factors[is_possible]):
        if facs > t_most:
            to_return.append(num)
            t_most = facs
    return t_most, to_return

def run_factor_batch(batch, mem):
    global most
    new_most, valid = is_antiprime(most, batch, test_factor_batch(batch, mem))
    if valid:
        antiprimes.extend(valid)
        most = new_most
        file_print(valid)

def file_print(print_obj, out_file='out.txt'):
    print(print_obj)
    if PRINT_FILE:
        with open(out_file, 'a') as f:
            print(print_obj, file=f)


gpu_mem = cuda.tools.DeviceMemoryPool()
for n in range(1, limit + 1, BATCH_SIZE):
    n_max = limit + 1 if limit < n + BATCH_SIZE else n + BATCH_SIZE
    run_factor_batch(np.arange(n, n_max, dtype=m_data_type), gpu_mem)

file_print(antiprimes)
file_print(time() - start_time)
