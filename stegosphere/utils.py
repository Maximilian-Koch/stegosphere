import numpy as np

def dtype_range(dtype):
    """
    Gets minimum/maximum value of numpy datatype.

    :param dtype: dtype to be checked
    :return: tuple, with minimum and maximum value.
    """
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise Exception('array dtype must be integer or float.')
    return info.min, info.max

def prng_indices(length, key):
    """
    Shuffles indices of array using PCG64DXSM.

    :param length: Length of array.
    :param key: Seed to use for the PRNG.
    :return: Shuffled indices.
    """
    if type(key)!=int:
        key = np.frombuffer(key.encode(), dtype=np.uint32)
    rng = np.random.Generator(np.random.PCG64DXSM(seed=key))
    indices = np.arange(length)
    rng.shuffle(indices)
    return indices

def generate_binary_payload(length):
    """
    Generate a binary payload.

    :param length: The length of the payload.
    :type length: int

    :return: payload
    :rtype: str
    """
    binary_array = np.random.randint(0, 2, size=length)
    return ''.join(map(str, binary_array))
