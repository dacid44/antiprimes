import pycuda.gpuarray as gpuarray
import pycuda.cumath as cumath
from pycuda.elementwise import ElementwiseKernel
from pycuda.compiler import SourceModule
import pycuda as cuda
import pycuda.autoinit
import numpy as np
from time import time


is_equal = ElementwiseKernel("unsigned int *x, unsigned int *y, bool *z", "z[i] = x[i] == y[i]", "is_equal")
modulo = ElementwiseKernel("unsigned int *x, unsigned int *y, unsigned int *z", "z[i] = x[i] % y[i]", "modulo")
is_div = ElementwiseKernel("unsigned int x, unsigned int *y, bool *z", "z[i] = (x % y[i]) == 0", "is_div")

'''
count_facs = SourceModule("""
    __global__ void count_facs(int x, unsigned int *y, int *z) {
        if (x % y[threadIdx.x]) {
            z[0] += 1;
        }
    }
""").get_function("count_facs")
'''

limit = 5000 # limit = int(input('Limit: '))
start_time = time()
with open('primes1.txt') as f:
    primes = np.fromiter(map(int, f.read().strip().split(',')), dtype=np.uint32)
p_gpu = gpuarray.to_gpu(primes)

antiprimes = []
most = 0
for x in gpuarray.arange(2, limit + 1, dtype=np.uint32):
    x_local = x.get()
    fac = [[], []]
    x_temp = x.copy()
    for prime in p_gpu:
        more = cumath.fmod(x_temp, prime).get() == np.uintc(0)
        more_app = more
        if more:
            power = 0
        while more:
            power += 1
            x_temp = x_temp / prime
            more = cumath.fmod(x_temp, prime).get() == np.uintc(0)
        if more_app:
            fac[0].append(prime.get())
            fac[1].append(power)
        if x_temp.get() <= np.uintc(1):
            break
    if not fac[0]:
        continue
    num_primes = len(fac[0])
    eq_gpu = gpuarray.to_gpu(np.empty(num_primes, dtype=np.bool))
    is_equal(gpuarray.to_gpu(np.fromiter(fac[0], dtype=np.uint32)), gpuarray.to_gpu(primes[:num_primes]), eq_gpu)
    if not np.all(eq_gpu.get()):
        continue
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
    powers = np.fromiter(fac[1], np.uint32)
    if not np.all(powers[:-1] >= powers[1:]):
        continue
    # Calculate number of factors
    sqrt = np.sqrt(x_local)
    tests = gpuarray.arange(1, np.uint(np.ceil(sqrt)), dtype=np.uint32)
    mod_gpu = gpuarray.empty_like(tests, dtype=np.bool)
    is_div(x_local, tests, mod_gpu)
    factors = np.count_nonzero(mod_gpu.get())
    factors *= 2
    if sqrt == np.round(sqrt):
        factors += 1
    if factors > most:
        antiprimes.append(int(x_local))
        most = factors
        print(x_local)
print(antiprimes)
print(time() - start_time)
