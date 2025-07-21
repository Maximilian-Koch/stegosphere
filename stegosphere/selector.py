"""
Index selection methods for data embedding.

Provides deterministic and probabilistic strategies for selecting
embedding locations based on cost metrics or pseudorandom generators.

Includes:
- Greedy selection by cost
- PRNG-based shuffling and sampling (PCG64DXSM)
- Weighted random sampling from inverse-costs or arbitrary weights
"""

import numpy as np

__all__ = ['greedy_indices','prng_indices','prng_indices_choice',
            'weighted_greedy_prng_indices', 'inverse_cost_greedy_prng_indices']

def greedy_indices(costs, n=-1):
    """
    Sorts indices by cost and returns the `n` smallest costs's indices.

    Args:
        costs (numpy.ndarray): Cost-function output
        n (optional): Return `n` indices, defaults to -1 for all indices
    
    Returns:
        numpy.ndarray: Array of indices
    """
    flat = costs.ravel()
    small_idx = np.argpartition(flat, n)[:(n if n > 0 else None)]
    return small_idx[np.argsort(flat[small_idx])]


#either generates indices of array size with array size as maximum index using rng.shuffle,
#or chooses a specified  number (length) of indices using rng.choice with array size as upper bound.
#In almost all scenarios the second one is more memory efficient (and time  efficient),
#but the needed length not necessarily known.
#these functions will be adapted in future versions.
def prng_indices(size, key):
    """
    Shuffles indices of array using PCG64DXSM.

    Args:
        size: size of array to be shuffled
        key: seed to use for PRNG
    
    Returns:
        numpy.ndarray: Shuffled indices
    """
    if type(key)!=int:
        key = np.frombuffer(key.encode(), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed=key))
    indices = np.arange(size)
    rng.shuffle(indices)
    return indices

def prng_indices_choice(size, key, length):
    """
    Randomly selects `length` indices using a `key` as a seed, with `size` as upper-bound.
    More efficient for most scenarios than `prng_indices`. Will be merged in the future.
    """
    if not isinstance(key, int):
        key = np.frombuffer(key.encode(), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed=key))
    return rng.choice(size, size=length, replace=False)

def weighted_greedy_prng_indices(max_index, weights, n, key):
    """
    Samples `n` distinct indices using a PRNG with a specified weight distribution.

    Args:
        max_index: Upper bound of indices used
        weights (numpy.ndarray): Weights used for sampling
        n: Number of indices to pick
        key: Seed for PRNG
    
    Returns:
        numpy.ndarray: Flat array of indices
    """

    if not isinstance(key, int):
        key = np.frombuffer(key.encode(), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed=key))

    p = weights / weights.sum()
    return rng.choice(max_index,size=n, replace=False, p=p)

def inverse_cost_greedy_prng_indices(costs, n, key):
    """
    Samples `n` distinct indices from costs, with probability 
    proportional to 1/costs (lower‐cost positions more likely).

    Args:
        costs (numpy.ndarray): Array of embedding costs
        n: Number of indices to pick
        key: Seed for PRNG
    Returns:
        numpy.ndarray: Flat array of indices
    """
    costs = costs.copy().ravel()
    w = 1.0 / costs
    return weighted_greedy_prng_indices(len(costs),w,n,key)