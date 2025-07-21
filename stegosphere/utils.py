"""
Module for various utilities
"""

import numpy as np

__all__ = ['dtype_range','generate_binary_payload']

def dtype_range(dtype):
    """
    Gets minimum/maximum value of numpy datatype.

    Args:
        dtype: dtype to be checked
    Returns:
        tuple: Containing minimum and maximum value

    Raises:
        TypeError: if dtype is not `numpy.integer` or `numpy.floating`
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise TypeError('array dtype must be integer or float.')
    return info.min, info.max

#use PRNG here as also in selector.py
def generate_binary_payload(length):
    """
    Generates a binary payload.

    Args:
        length: Length of the payload
    Returns:
        str: Payload
    """
    binary_array = np.random.randint(0, 2, size=length)
    return ''.join(map(str, binary_array))
