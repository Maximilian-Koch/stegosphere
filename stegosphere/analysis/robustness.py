#robustness attacks

import numpy as np

def random_bit_attack(array, ratio=0.01):
    """
    Change given ratio of array elements by +1 or -1, randomly.
    """
    array = array.copy()
    indices = np.random.choice(array.size, int(array.size * ratio), replace=False)
    perturbations = np.random.choice([-1, 1], size=len(indices))
    #not robust to underflow/overflow 
    array.flat[indices] += perturbations
    return array
