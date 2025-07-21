"""
Implements WOW cost, as presented in

V. Holub and J. Fridrich, "Designing steganographic distortion using directional filters,"
2012 IEEE International Workshop on Information Forensics and Security (WIFS), Costa Adeje, Spain, 2012, pp. 234-239,
doi: 10.1109/WIFS.2012.6412655.
"""

import numpy as np
from scipy.signal import convolve2d

def wow_cost(cover, alpha=10):
    """
    Calculates the WOW cost of a given cover.

    Args:
        cover (numpy.ndarray): Cover to be evaluated
        alpha (float, optional): alpha to be used
    Returns:
        numpy.ndarray: cost map
    """
    #directional filters
    kernels = [
        np.array([[1, -1]]),
        np.array([[1], [-1]]),
        np.array([[1, -2, 1]]),
        np.array([[1], [-2], [1]]),
        np.array([[1, 0, -1]]),
        np.array([[1], [0], [-1]])]
    R = sum(np.abs(convolve2d(cover, k, mode='same')) for k in kernels)
    #local smoothing
    R = convolve2d(R, np.ones((3,3))/9, mode='same')
    return 1.0 / (R + alpha)