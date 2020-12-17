import multiprocessing
import threading
import datetime

from math import sqrt

class antiprimes:
    def __init__(self, filename, threads):
        self.print = True
        self.threads = threads
        with open(filename, 'r') as f:
            for line in f:
                self.primes = tuple(map(lambda n: int(n), line[:-1].split(',')))

    def log(self, line):
        with open('log.txt', 'a') as f:
            f.write(line + '\n')
            f.close()
        if self.print:
            print(line)

    def prime_fac(self, num):
        fac = {}
        for prime in self.primes:
            more = num % prime == 0
            if more:
                fac[prime] = 0
            while more:
                fac[prime] += 1
                num /= prime
                more = num % prime == 0
            if num <= 1:
                break
        return fac

    def perf_sqrt(self, num, pFac):
        perf = True
        for factor, power in pFac.items():
            if power % 2 == 1:
                perf = False
                break
        if perf:
            return (round(sqrt(num)), True)
        else:
            return (sqrt(num), False)

    def num_factors(self, num):
        pFac = self.prime_fac(num)
        
        if len(pFac) > 0:
            items = tuple(pFac.items())
            val = 0
            for factor in reversed(self.primes[:self.primes.index(items[len(items) - 1][0])]):
                if factor not in pFac.keys():
                    return 0
                if pFac[factor] >= val:
                    val = pFac[factor]
                else:
                    return 0
        else:
            return 0
        
        root = self.perf_sqrt(num, pFac)
        count = 0
        if root[1]:
            limit = root[0]
        else:
            limit = int(root[0]) + 1
        for i in range(1, limit):
            '''
            isFac = True
            for factor, power in self.prime_fac(i).items():
                if factor not in pFac.keys():
                    isFac = False
                    break
                elif power > pFac[factor]:
                    isFac = False
                    break
            '''
            if num % i == 0:
                count += 1
        count *= 2
        if root[1]:
            count += 1
        return count

    def antiprimes(self, limit):
        antiprimes = []
        most = 0
        for i in range(1, limit + 1):
            factors = self.num_factors(i)
            if factors > most:
                most = factors
                antiprimes.append(i)
                print(i)
        return antiprimes

    def aprime_process(self, start, limit, resultQueue, printQueue):
        printQueue.put('process ' + multiprocessing.current_process().name + ' started')
        antiprimes = []
        most = self.num_factors(start)
        for i in range(start, limit):
            factors = self.num_factors(i)
            if factors > most:
                most = factors
                antiprimes.append(i)
                printQueue.put(multiprocessing.current_process().name + ': ' + str(i))
        resultQueue.put(antiprimes)
        printQueue.put('process ' + multiprocessing.current_process().name + ' finished')

    def printThread(self, printQueue):
        #print('printThread started')
        with open('log.txt', 'w') as f:
            f.write('printThread started\n')
            f.close()
        while True:
            toPrint = printQueue.get()
            if toPrint == 'stop':
                break
            else:
                self.log(str(toPrint))
        self.log('printThread stopped')
        f.close()

    def mpAPrimes(self, limit):
        limit += 1
        cutoffs = [1]
        for i in range(1, self.threads):
            cutoffs.append(int(sqrt(i*(limit**2)/self.threads)))
        cutoffs.append(limit)

        resultQueue = multiprocessing.SimpleQueue()
        printQueue = multiprocessing.SimpleQueue()

        printThread = threading.Thread(target = self.printThread, args = (printQueue,), name = 'printThread')
        printThread.start()

        for i in range(1, self.threads):
            printQueue.put('cutoff ' + str(i) + ': ' + str(cutoffs[i]))

        jobs = []
        for i in range(self.threads):
            jobs.append(multiprocessing.Process(target = self.aprime_process, args = (cutoffs[i], cutoffs[i + 1], resultQueue, printQueue), name = 'p' + str(i)))

        for job in jobs:
            job.start()

        aPrimeList = []
        for i in range(self.threads):
            aPrimeList += resultQueue.get()
            printQueue.put('got result ' + str(i))
        for job in jobs:
            job.join()

        printQueue.put(aPrimeList)
        aPrimeList.sort()
        antiprimes = []
        most = 0
        for num in aPrimeList:
            factors = self.num_factors(num)
            if factors > most:
                most = factors
                antiprimes.append(num)
                printQueue.put(num)
        printQueue.put(antiprimes)

        printQueue.put('stop')
        printThread.join()
        
        return antiprimes

if __name__ == '__main__':
    a = antiprimes('primes1.txt', int(input('Number of threads: ')))
    startTime = datetime.datetime.now()
    toPrint = a.mpAPrimes(int(input('Limit: ')))
    endTime = datetime.datetime.now()
    timeStr = 'Time: ' + str((endTime - startTime).total_seconds())
    a.log(timeStr)
