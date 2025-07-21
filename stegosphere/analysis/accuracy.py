"""
Accuracy metrics for comparing original and extracted binary payloads.
"""

import numpy as np

__all__ = ['hamming_distance','bit_error_rate','normalized_correlation']

def hamming_distance(before, after):
    """
    Compute the Hamming distance between two binary strings or arrays.

    Args:
        before (str | list | numpy.ndarray): Original binary sequence.
        after (str | list | numpy.ndarray): Modified binary sequence.

    Returns:
        int: Number of differing bits.

    Raises:
        ValueError: If inputs are not the same length.
    """
    if len(before) != len(after):
        raise ValueError("Inputs must be of equal length")

    before = np.array(list(before)) if isinstance(before, str) else np.asarray(before)
    after = np.array(list(after)) if isinstance(after, str) else np.asarray(after)

    return np.sum(before != after)

def bit_error_rate(before, after):
    """
    Compute the bit error rate between two binary sequences.

    Args:
        before (str | list | numpy.ndarray): Original binary sequence.
        after (str | list | numpy.ndarray): Modified binary sequence.

    Returns:
        float: Proportion of bits that differ.

    Raises:
        ValueError: If inputs are not the same length.
    """
    if len(before) != len(after):
        raise ValueError("Inputs must be of equal length")
    if len(before) == 0:
        return 0.0
    return hamming_distance(before, after) / len(before)

def normalized_correlation(before, after):
    """
    Compute the normalized correlation between two sequences.

    Args:
        before: Original numerical sequence.
        after: Modified numerical sequence.

    Returns:
        float: Normalized correlation coefficient. Returns 0 if denominator is zero.

    Raises:
        ValueError: If inputs are not the same length.
    """
    if len(before) != len(after):
        raise ValueError("Inputs must be of equal length")

    before = np.asarray(before)
    after = np.asarray(after)

    numerator = np.sum(before * after)
    denominator = np.sum(before ** 2)
    return numerator / denominator if denominator != 0 else 0.0
