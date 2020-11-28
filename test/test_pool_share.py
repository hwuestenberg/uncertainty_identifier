import multiprocessing
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
from multiprocessing import Process

from memory_profiler import profile
from psutil import virtual_memory

import numpy as np
import time
import random
import sys

import gc

from uncert_ident.utilities import time_decorator



#
# Functions used by test code
#

def calculate(func, args):
    result = func(*args)
    return '%s says that %s%s = %s' % (
        multiprocessing.current_process().name,
        func.__name__, args, result
        )

def calculatestar(args):
    return calculate(*args)

def mul(a, b):
    time.sleep(0.5 * random.random())
    return a * b

def plus(a, b):
    time.sleep(0.5 * random.random())
    return a + b

def f(x):
    return 1.0 / (x - 5.0)

def pow3(x):
    return x ** 3

def noop(x):
    pass

def countdown(x):
    i = x
    while i > 0:
        i -= 1

def countdown_huge(*args):
    smz_name, smz_size, smz_type = args

    # Allocate shared memory
    existing_smz = shared_memory.SharedMemory(name=smz_name)
    z = np.ndarray(smz_size, dtype=smz_type, buffer=existing_smz.buf)

    i = z[-1]
    if z[-1] > 1e7:
        i = 1e7
    print("Counting down...")
    while i > 0:
        i -= 1

    existing_smz.close()

    return "Boom"

def countdown_huge_smm(xx, ret):
    i = xx[-1]
    if xx[-1] > 1e7:
        i = 1e7
    print("Counting down...")
    while i > 0:
        i -= 1

    ret.append(1)
    # return 1


#
# Test code
#







@time_decorator
def test(proc):
    print("In test()....")
    print(f"\nCPUs:\t{multiprocessing.cpu_count()}")
    print('Creating pool with %d processes\n' % proc)

    with multiprocessing.Pool(proc) as pool:

        TASKS = [(mul, (i, 7)) for i in range(10)] + \
                [(plus, (i, 8)) for i in range(10)]
        # TASKS = [((countdown, (i,))) for i in [1e8]*8]
        #
        # results = [pool.apply_async(calculate, t) for t in TASKS]
        # imap_it = pool.imap(calculatestar, TASKS)
        # imap_unordered_it = pool.imap_unordered(calculatestar, TASKS)

        # print('Ordered results using pool.apply_async():')
        # for r in results:
        #     print('\t', r.get())
        # print()
        #
        # print('Ordered results using pool.imap():')
        # for x in imap_it:
        #     print('\t', x)
        # print()
        #
        # print('Unordered results using pool.imap_unordered():')
        # for x in imap_unordered_it:
        #     print('\t', x)
        # print()

        print('Ordered results using pool.map() --- will block till complete:')
        for x in pool.map(calculatestar, TASKS):
            print('\t', x)
        print()



@time_decorator
@profile
def test2(harr_name, harr_size, harr_type, proc, n_tasks):
    print("In test2()....")
    print(f"\nCPUs:\t{multiprocessing.cpu_count()}")
    print('Creating pool with %d processes\n' % proc)

    # TASKS = [(mul, (i, 7)) for i in range(10)] + \
    #         [(plus, (i, 8)) for i in range(10)]
    # TASKS = [((countdown, (i,))) for i in [1e8]*8]
    TASKS = [((countdown_huge, ([harr_name, harr_size, harr_type]))) for i in [1] * n_tasks]


    # with multiprocessing.Pool(PROCESSES) as new_pool:
    #     map_rslt = new_pool.map(calculatestar, TASKS)
    new_pool = multiprocessing.Pool(proc)
    map_rslt = new_pool.map(calculatestar, TASKS)
    new_pool.close()

    print("Results with pool.map()")
    [print(r) for r in map_rslt]


    return 1




@time_decorator
@profile
def test_smm(harr, proc, n_tasks):
    print("In test_smm()....")
    print(f"\nCPUs:\t{multiprocessing.cpu_count()}")
    print('Creating pool with %d processes\n' % proc)
    print('For %d tasks\n' % n_tasks)

    # TASKS = [(mul, (i, 7)) for i in range(10)] + \
    #         [(plus, (i, 8)) for i in range(10)]
    # TASKS = [((countdown, (i,))) for i in [1e8]*8]
    TASKS = [((countdown_huge, ([harr_name, harr_size, harr_type]))) for i in [1] * n_tasks]


    # with multiprocessing.Pool(PROCESSES) as new_pool:
    #     map_rslt = new_pool.map(calculatestar, TASKS)
    new_pool = multiprocessing.Pool(proc)
    map_rslt = new_pool.map(calculatestar, TASKS)
    new_pool.close()

    print("Results with pool.map()")
    [print(r) for r in map_rslt]


    return 1





@profile
def allocate_share(array):
    sm_array = shared_memory.SharedMemory(create=True, size=array.nbytes)
    arr = np.ndarray(array.shape, dtype=array.dtype, buffer=sm_array.buf)
    arr[:] = array
    sm_size = array.shape
    sm_type = array.dtype
    del array
    gc.collect()

    return sm_array, sm_size, sm_type


@profile
def free_share(sm_array):
    sm_array.close()
    sm_array.unlink()
    return 1



if __name__ == '__main__':
    processes = 4

    # # Allocate huge array
    # print(f"Before allocate z\tMemory used:{virtual_memory().percent}\tor {virtual_memory().used / 1e9}Gb\tof total:{virtual_memory().total / 1e9}Gb\tand active:{virtual_memory().active / 1e9}Gb")
    # z = np.arange(9e8)  # 1e9 = 8Gb
    # print(f"Nbytes of z:\t{z.nbytes/1e9:3.2f}Gb")
    # print(f"After allocate z\tMemory used:{virtual_memory().percent}\tor {virtual_memory().used / 1e9}Gb\tof total:{virtual_memory().total / 1e9}Gb\tand active:{virtual_memory().active / 1e9}Gb")
    # smz, smz_size, smz_type = allocate_share(z)
    # print(f"After allocate_share\tMemory used:{virtual_memory().percent}\tor {virtual_memory().used/1e9}Gb\tof total:{virtual_memory().total/1e9}Gb\tand active:{virtual_memory().active/1e9}Gb")
    #
    # # test(smz, processes)
    # test2(smz.name, smz_size, smz_type, processes, 12)
    #
    #
    # # Free shared memory
    # free_share(smz)
    # print(f"After free_share\tMemory used:{virtual_memory().percent}\tor {virtual_memory().used / 1e9}Gb\tof total:{virtual_memory().total / 1e9}Gb\tand active:{virtual_memory().active / 1e9}Gb")
    # del z
    # print(f"After del z\tMemory used:{virtual_memory().percent}\tor {virtual_memory().used/1e9}Gb\tof total:{virtual_memory().total/1e9}Gb\tand active:{virtual_memory().active/1e9}Gb")

    array = np.arange(9e8)
    with SharedMemoryManager() as smm:
        sm_array = smm.SharedMemory(size=array.nbytes)
        arr = np.ndarray(array.shape, dtype=array.dtype, buffer=sm_array.buf, order='F')
        arr[:] = array
        sm_size = array.shape
        sm_type = array.dtype

        print('z contiguous: ', array.flags['F_CONTIGUOUS'], flush=True)
        print('zz contiguous: ', arr.flags['F_CONTIGUOUS'], flush=True)

        del array
        gc.collect()

        procs = [None] * processes
        returns = [[] for i in range(processes)]
        for j in range(processes):
            procs[j] = Process(target=calculatestar, args=[[countdown_huge_smm, (arr, returns[j])]])
            procs[j].start()

        for j in range(processes):
            procs[j].join()

        print("Returns are:")
        print(returns)


    print("Finished")

