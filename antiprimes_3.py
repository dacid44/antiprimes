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
FACTOR_BATCH_SIZE = 50

limit = 100000 # limit = int(input('Limit: '))
start_time = time()
with open('primes1.txt') as f:
    primes = np.fromiter(map(int, f.read().strip().split(',')), dtype=np.uint32)

@njit
def prime_index(n):
    for idx, val in np.ndenumerate(primes):
        if val >= n:
            return idx[0]

@njit(parallel=True)
def fill_rows(start, stop, width):
    to_return = np.empty((stop - start, width), dtype=np.int32)
    for n in prange(start, stop):
        to_return[n - start] = n
    return to_return

@njit(parallel=True)
def get_powers(mat, primes_arr):
    for n in prange(mat.shape[0]):
        powers = np.zeros(primes_arr.size, dtype=np.uint32)
        while True:
            is_div = mat[n] % primes_arr == 0
            powers += is_div
            if not np.any(is_div):
                break
            mat[n][is_div] //= primes[is_div]
        mat[n] = powers

@njit(parallel=True)
def get_candidates(mat, start, stop):
    numbers = np.arange(start, stop, 1, np.uint32)
    is_cand = np.empty_like(numbers, dtype=np.bool_)
    for n in prange(is_cand.size):
        is_cand[n] = np.all(mat[n,:-1] >= mat[n,1:])
    return numbers[is_cand]

cand_queue = deque()
most = np.array([0], dtype=np.uint32)
antiprimes = []

def get_next_batch():
    to_return = np.empty(len(cand_queue) if len(cand_queue) < FACTOR_BATCH_SIZE else FACTOR_BATCH_SIZE, dtype=np.uint32)
    for i in range(to_return.size):
        to_return[i] = cand_queue.popleft()
    return to_return

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

def run_factor_batch(end=False):
    global most
    while (len(cand_queue) >= FACTOR_BATCH_SIZE) if not end else (len(cand_queue) > 0):
        batch = get_next_batch()
        num_factors = test_factor_batch(batch)
        for i in range(batch.size):
            if num_factors[i] > most:
                antiprimes.append(batch[i])
                most[0] = num_factors[i]
                print(batch[i])

for n in range(1, limit + 1, BATCH_SIZE):
    n_max = limit + 1 if limit < n + BATCH_SIZE else n + BATCH_SIZE
    n_primes = primes[:prime_index(n_max)]
    powers_mat = fill_rows(n, n_max, n_primes.size)
    get_powers(powers_mat, n_primes)
    candidates = get_candidates(powers_mat, n, n_max)
    cand_queue.extend(candidates)
    run_factor_batch()

run_factor_batch(True)
print(antiprimes)
print(time() - start_time)
