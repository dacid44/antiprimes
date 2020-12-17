import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
import pycuda as cuda
import pycuda.autoinit
import numpy as np
from time import time


# is_equal = ElementwiseKernel("unsigned int *x, unsigned int *y, bool *z", "z[i] = x[i] == y[i]", "is_equal")
# is_equal = ElementwiseKernel("bool *z", "z[i] = 0", "equality_checker")
eq_checker = ElementwiseKernel(
        "int *x, int *y, int *z",
        "z[i] = x[i] == y[i]",
        "equality_checker")
# modulo = ElementwiseKernel("int *x, int *y, int *z", "z[i] = x[i] % y[i]", "modulo")

eq_gpu = gpuarray.to_gpu(np.empty(num_primes, dtype=np.int32))
a_gpu = gpuarray.arange(5, dtype=np.int32)
b_gpu = gpuarray.arange(1, 6, dtype=np.int32)
eq_checker(a_gpu, b_gpu, eq_gpu)

limit = 100 # limit = int(input('Limit: '))
start_time = time()
with open('primes1.txt') as f:
    primes = np.fromiter(map(int, f.read().strip().split(',')), dtype=np.uint32)
p_gpu = gpuarray.to_gpu(primes)

antiprimes = []
most = 0
for x in gpuarray.arange(2, limit + 1, dtype=np.uint32):
    print(x)
    fac = [[], []]
    x_temp = x.copy()
    for prime in p_gpu:
        more = cumath.fmod(x_temp, prime).get() == np.uintc(0)
        if more:
            power = 0
        while more:
            power += 1
            x_temp = x_temp / prime
            more = cumath.fmod(x_temp, prime).get() == np.uintc(0)
        if more:
            fac[0].append(prime)
            fac[1].append(power)
        if x_temp.get() <= np.uintc(1):
            break
    if not fac:
        continue
    num_primes = len(fac[0])
    eq_gpu = gpuarray.to_gpu(np.empty(num_primes, dtype=np.int32))
    a_gpu = gpuarray.arange(5, dtype=np.int32)
    b_gpu = gpuarray.arange(1, 6, dtype=np.int32)
    eq_checker(a_gpu, b_gpu, eq_gpu)
    # is_equal(eq_gpu)
    # eq_checker(gpuarray.to_gpu(np.fromiter(fac[0], dtype=np.uint32)), gpuarray.to_gpu(primes[:num_primes]), eq_gpu)
    # is_equal(gpuarray.to_gpu(np.fromiter(fac[0], dtype=np.uint32)), gpuarray.to_gpu(primes[:num_primes]), eq_gpu)
    '''
    if not np.all(eq_gpu.get()):
        continue
    '''
    '''
    iter_fac = iter(fac)
    is_invalid = False
    for i in range(len(fac)):
        if next(iter_fac) != primes[i]:
            is_invalid = True
            break
    if is_invalid:
        continue
    '''
    powers = np.fromiter(fac.values(), np.uint32)
    if not np.all(powers[:-1] >= powers[1:]):
        continue
    # Calculate number of factors
    sqrt = np.sqrt(x)
    tests = np.arange(1, np.uint(np.ceil(sqrt)))
    factors = tests.size - np.count_nonzero(np.mod(x, tests))
    factors *= 2
    if sqrt == np.round(sqrt):
        factors += 1
    if factors > most:
        antiprimes.append(x)
        most = factors
        print(x)
print(antiprimes)
print(time() - start_time)
