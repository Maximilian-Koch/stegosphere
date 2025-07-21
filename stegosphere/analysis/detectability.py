"""
A module to detect the likelihood of steganography
"""

import numpy as np
import random

__all__ = ['random_detector', 'uniformity_detector', 'sp']

def random_detector(input_data):
    """
    Randomly decide presence of steganography (for testing).

    Args:
        input_data: Any input; ignored in decision.

    Returns:
        int: 0 or 1, chosen uniformly at random.
    """
    return random.randint(0, 1)

def uniformity_detector(array):
    """
    Detect likelihood of LSB steganography in an RGB image by uniformity of LSBs.

    Args:
        array (array-like): H×W×3 RGB image array of integers.

    Returns:
        float: Score ∈ [0,1], where higher means more uniform LSB distribution
               (i.e. more likely stego).

    Raises:
        ValueError: If input is not a 3D array with third dimension size 3.
    """
    arr = np.asarray(array)
    if arr.ndim != 3 or arr.shape[2] != 3:
        raise ValueError("Input must be an H×W×3 RGB image array.")

    #LSB of all channels
    lsb_all = np.mod(arr, 2).ravel()

    count_zeros = np.count_nonzero(lsb_all == 0)
    count_ones = np.count_nonzero(lsb_all == 1)
    total = count_zeros + count_ones

    if total == 0:
        return 0.0

    zero_ratio = count_zeros / total
    one_ratio = count_ones / total
    return 1.0 - abs(zero_ratio - one_ratio)


def sp(X):
    """
    Estimate embedding rate (β̂) via Sample Pairs Analysis (Dumitrescu et al. 2002).

    Args:
        X (array-like): 2D array of integer pixel values.

    Returns:
        float: Estimated embedding rate β̂ ∈ [0,1], or -1 if undetermined.

    Raises:
        ValueError: If input is not a 2D array with at least 2 columns.
    """
    arr = np.asarray(X, dtype=float)
    if arr.ndim != 2 or arr.shape[1] < 2:
        raise ValueError("Input must be a 2D array with at least two columns.")

    #adjacent pairrs
    u = arr[:, :-1].ravel()
    v = arr[:,  1:].ravel()
    n_pairs = u.size

    Xc = np.count_nonzero(((v % 2 == 0) & (u < v)) |
        ((v % 2 == 1) & (u > v)))
    Zc = np.count_nonzero(u == v)
    Wc = np.count_nonzero((np.floor_divide(u, 2) == np.floor_divide(v, 2)) & (u != v))
    Vc = n_pairs - (Xc + Zc + Wc)

    a = (Wc + Zc) / 2.0
    b = 2 * Xc - n_pairs
    c = Vc + Wc - Xc

    D = b**2 - 4*a*c

    if a <= 0 or D < 0:
        return -1.0

    sqrtD = np.sqrt(D)
    p1 = (-b + sqrtD) / (2 * a)
    p2 = (-b -sqrtD) / (2 * a)

    #choose root in [0,1]
    candidates = [p.real for p in (p1, p2) if 0 <= p.real <= 1]
    return min(candidates) if candidates else -1.0