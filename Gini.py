 
import numpy as np
 
 
def gini(x):
    """
    Compute the Gini coefficient of a 1-D array of incomes x.
 
    Uses formula:
        G = (2 * sum(rank_i * x_i) - (N + 1) * sum(x)) / (N * sum(x))
    where incomes are sorted ascending and rank_i = 1, 2, ..., N.
    This is algebraically equivalent to the standard Lorenz-curve
    definition of the Gini coefficient, but simpler to implement directly.
 
    Parameters
    ----------
    x : array-like
        A 1-D array (or list) of incomes. Can be a single age group,
        a full pooled sample, or any other collection of income values.
 
    Returns
    -------
    float
        The Gini coefficient, between 0 (perfect equality) and 1
        (perfect inequality).
    """
    x = np.sort(np.asarray(x, dtype=float))  # sort incomes ascending
    N = len(x)
    index = np.arange(1, N + 1)              # ranks 1, 2, ..., N
    return (2 * np.sum(index * x) - (N + 1) * np.sum(x)) / (N * np.sum(x))