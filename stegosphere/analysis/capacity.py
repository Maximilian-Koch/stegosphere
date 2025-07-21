"""
Measuring how much data can be stored within a cover.

In future versions, this should include heuristics using the `max_capacity_estimator`
"""

import numpy as np

__all__ = ['bits_per_value', 'payload_ratio', 'bits_per_value_used', 'max_capacity_estimator']

def max_capacity_estimator(*args):
    raise NotImplementedError

def bits_per_value(array, max_capacity):
    """
    Compute bits per host‐value given total capacity.

    Args:
        array (numpy.ndarray): Host array.
        max_capacity (int | float): Total embedding capacity in bits.
    Returns:
        float: Bits of payload per array element.
    """
    n = np.asarray(array).size

    return max_capacity / n

def payload_ratio(payload_size, max_capacity):
    """
    Compute payload-to-capacity ratio.

    Args:
        payload_size (int | float): Size of embedded payload in bits.
        max_capacity (int | float): Maximum capacity in bits.
    Returns:
        float: Ratio payload_size / max_capacity.
    """
    return payload_size / max_capacity

def bits_per_value_used(array, payload_size):
    """
    Compute actually used bits per host‐value.

    Args:
        array (numpy.ndarray): Host array.
        payload_size (int | float): Size of embedded payload in bits.
    Returns:
        float: Payload bits per array element.
    """
    n = np.asarray(array).size
    return payload_size / n