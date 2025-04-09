import numpy as np

def hamming_distance(before, after):
    """
    Compute Hamming Distance (HD)
    
    :param before: Binary string or NumPy array
    :param after: Binary string or NumPy array
    :return: Hamming Distance (integer)
    """
    assert len(before) == len(after), "Inputs must be of equal length"

    if isinstance(before, str) and isinstance(after, str):
        before = np.array(list(before))
        after = np.array(list(after))

    return np.sum(before != after)

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
