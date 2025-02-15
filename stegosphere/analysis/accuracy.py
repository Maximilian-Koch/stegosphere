import numpy as np

def hamming_distance(before, after):
    """
    Compute Hamming Distance (HD)

    :param before:
    :param after:

    :return: Hamming Distance
    """
    assert len(before) == len(after)
    return np.sum(np.array(before) != np.array(after))

def bit_error_rate(before, after):
    """
    Compute Bit Error Rate (BER)
    """
    assert len(before) == len(after)
    if before == '':
        return 0
    return hamming_distance(before, after) / len(before)

def normalized_correlation(before, after):
    """
    Compute Normalized Correlation (NC)
    """
    assert len(before) == len(after)
    before = np.array(before)
    after = np.array(after)
    numerator = np.sum(before * after)
    denominator = np.sum(before ** 2)
    return numerator / denominator if denominator != 0 else 0
