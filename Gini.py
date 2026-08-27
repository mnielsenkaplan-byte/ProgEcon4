import numpy as np
 
 
def gini(x):

    x = np.sort(np.asarray(x, dtype=float))  #sort incomes, ascending
    N = len(x)
    index = np.arange(1, N + 1)              #ranks 1, 2, ..., N
    return (2 * np.sum(index * x) - (N + 1) * np.sum(x)) / (N * np.sum(x))