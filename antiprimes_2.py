import numpy as np
from time import time

limit = 10000 # limit = int(input('Limit: '))
start_time = time()
with open('primes1.txt') as f:
    primes = np.fromiter(map(int, f.read().strip().split(',')), dtype=np.uint32)

antiprimes = []
most = 0
for x in np.arange(2, limit + 1):
    fac = {}
    x_temp = x
    for prime in primes:
        more = np.mod(x_temp, prime) == np.uintc(0)
        if more:
            fac[prime] = 0
        while more:
            fac[prime] += 1
            x_temp = np.floor_divide(x_temp, prime)
            more = np.mod(x_temp, prime) == np.uintc(0)
        if x_temp <= np.uintc(1):
            break
    if not fac:
        continue
    num_primes = len(fac)
    if not np.array_equal(np.fromiter(fac, np.uint32), primes[:num_primes]):
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
    powers = np.fromiter(fac.values(), np.uint32)
    if not np.all(powers[:-1] >= powers[1:]):
        continue
    # Calculate number of factors
    sqrt = np.sqrt(x)
    tests = np.arange(1, np.int(np.ceil(sqrt)))
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
