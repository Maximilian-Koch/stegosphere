import numpy as np


"""
payload_size: size of payload in bits embedded in the array
max_capacity: maximum size of payload that can be embedded
"""

def max_capacity_estimator(array, method):
    pass

def bits_per_value(array, max_capacity):
    return max_capacity / array.size

def payload_ratio(payload_size, max_capacity):
    return payload_size / max_capacity

def bits_per_value_used(array, payload_size):
    return payload_size / array.size
