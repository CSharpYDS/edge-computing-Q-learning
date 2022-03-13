import numpy as np
from params import *
PROC_MIN = int(1.10 * 1)
PROC_MAX = int(1.20 * 3)

PROC_RNG = np.arange(PROC_MIN, PROC_MAX, step = 1, dtype = np.int32)
PROC_RNG_L = len(PROC_RNG)

def multoss(p_vec):
    return (np.random.rand() > np.cumsum(p_vec)).argmin()

def genHeavyTailDist(size):     #e.g. [0, 0, ... 1, 1]
    mid_size = size - size//6
    arr_1 = 0.1*np.random.rand(mid_size).astype(np.float64)
    arr_2 = 0.6+0.1*np.random.rand(size-mid_size).astype(np.float64)
    arr = np.sort( np.concatenate((arr_1, arr_2)) )
    return (arr / np.sum(arr))

def genHeavyHeadDist(size):     #e.g. [1, 1, ... 0, 0]
    arr = genHeavyTailDist(size)
    return arr[::-1]

def genProcessingParameter(redo = False):
    global PROC_RNG, PROC_RNG_L
    if redo:
        PROC_RNG = np.arange(PROC_MIN, PROC_MAX, step = 1, dtype = np.int32)
    params = np.zeros((N_SERVER, N_JOB), dtype = np.int32)
    for j in range(N_JOB):
        for m in range(N_SERVER):
            _tmp_dist = genHeavyHeadDist(PROC_RNG_L)
            params[m, j] = PROC_RNG[multoss(_tmp_dist)]
    return params

print(genProcessingParameter())
print(np.random.poisson(2,10))


x = np.random.random((3,2))
y = np.random.random((3,2))
z = np.random.random((1,2))
print(x)
print(y)
print(z)
k = np.concatenate([x,y,z])
print(k.shape)

import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

x = [1,2,3]
y = [1,2,3]
plt.figure()
plt.subplot(221)
plt.plot(x, y,'r', label='loss')

plt.subplot(222)
plt.plot(x, y,'r', label='loss')

plt.subplot(223)
plt.plot(x, y,'r', label='loss')

plt.subplot(224)
plt.plot(x, y,'r', label='loss')

plt.show()
plt.close()