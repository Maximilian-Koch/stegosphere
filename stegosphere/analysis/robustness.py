#Measures accuracy of embedding and robustness attacks

import numpy as np

def difference_count(array_1, array_2, absolute=False):
    array_1 = array_1.astype(np.int64)
    array_2 = array_2.astype(np.int64)
    diff = array_1 - array_2
    if absolute is True:
        diff = abs(diff)
    nums, counts = np.unique(diff, return_counts=True)

    return nums, counts

def extract_accuracy(payload_1, payload_2):
    err = 0
    for c1, c2 in zip(payload_1, payload_2):
        if c1!=c2:
            err += 1
    if len(payload_2)>len(payload_1):
        err += len(payload_2)-len(payload_1)
    return 1- err/len(payload_2)
