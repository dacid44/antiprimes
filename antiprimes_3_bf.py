# import pycuda.gpuarray as gpuarray
# import pycuda.cumath as cumath
# from pycuda.elementwise import ElementwiseKernel
# from pycuda.compiler import SourceModule
# import pycuda as cuda
# import pycuda.autoinit
import numpy as np
from numba import njit, prange
from time import time
from collections import deque

BATCH_SIZE = 5000

limit = 5000000 # limit = int(input('Limit: '))
start_time = time()

@njit(parallel=True)
def fill_rows(start, stop, width):
    to_return = np.empty((stop - start, width), dtype=np.int32)
    for n in prange(start, stop):
        to_return[n - start] = n
    return to_return

most = np.uint32(0)
antiprimes = []

@njit(parallel=True)
def test_factor_batch(batch):
    sqrts = np.sqrt(batch)
    ceils = np.ceil(sqrts)
    test_mat = fill_rows(1, np.uint32(np.ceil(sqrts[-1])), batch.size)
    for n in prange(0, test_mat.shape[0]):
        test_mat[n] = batch % test_mat[n]
    num_factors = np.zeros(batch.size, dtype=np.uint32)
    for n in prange(0, batch.size):
        if batch[n] != 1:
            count_facs = test_mat[:ceils[n] - 1, n]
            num_factors[n] = count_facs.size - np.count_nonzero(count_facs)
            num_factors[n] *= 2
    num_factors += np.equal(sqrts, ceils)
    return num_factors

def run_factor_batch(batch):
    global most
    num_factors = test_factor_batch(batch)
    for i in range(batch.size):
        if num_factors[i] > most:
            antiprimes.append(batch[i])
            most = num_factors[i]
            print(batch[i])

for n in range(1, limit + 1, BATCH_SIZE):
    n_max = limit + 1 if limit < n + BATCH_SIZE else n + BATCH_SIZE
    run_factor_batch(np.arange(n, n_max, dtype=np.uint32))

print(antiprimes)
print(time() - start_time)
