"""
This is an adapted and generalised version of the Pixel Value Differencing method as proposed by
Wu, D. C., & Tsai, W. H. (2003).
A steganographic method for images by pixel-value differencing.
Pattern recognition letters, 24(9-10), 1613-1626.
"""
import warnings
import math

import numpy as np

from stegosphere.selector import prng_indices
from stegosphere import io
from stegosphere import payloader

def embed(array, payload, ranges=None, range_table=None, seed=None,
          framing=payloader.FixedHeader, framing_kwargs={}):
            
    """
    Embed a payload into a cover array using Value Differencing steganography.

    Args:
        array (numpy.ndarray): Cover object
        payload (str):  String of binary numbers
        ranges (list, optional): pre-defined ranges
        range_table (list, optional): pre-defined  range table
        seed (optional): Seed for PRNG indices
        framing (stegosphere.payloader.PayloadFrame | None): Type of framing used for payload
        framing_kwargs (dict): Specific arguments for payload framing

    Returns:
        numpy.ndarray: Array with embedded payload

    """
    payload = io.encode_binary_payload(payload, framing, framing_kwargs)
    array = array.copy()

    if array.ndim == 1:
        array = array[:, None]  # shape (N,) → (N,1)
    
    spatial_dim = array.ndim - 1
    channel_dim = array.shape[-1]

    if range_table is None:
        if ranges is None:
            ranges =  define_range()
        range_table = define_range_table(ranges)

    value_pairs = _get_pairs(array, spatial_dim, seed)
    payload_index = 0
    max_len = len(payload)

    for (v1_coords, v2_coords) in value_pairs:
        if payload_index >= max_len:
            break
        for channel in range(channel_dim):
            idx1 = v1_coords + (channel,)
            idx2 = v2_coords + (channel,)
            v1 = int(array[idx1])
            v2 = int(array[idx2])
            d = v2-v1
            abs_d = abs(d)

            try:
                bits_to_hide, l_i, u_i = range_table[abs_d]
            except Exception:
                continue  # Skip if difference is out of range

            if bits_to_hide>0:
                #Test boundaries:
                if d%2!=0:
                    v1_test = v1 - math.ceil((u_i-d)/2)
                    v2_test = v2 + math.floor((u_i-d)/2)
                if d%2==0:
                    v1_test = v1 - math.floor((u_i-d)/2)
                    v2_test = v2 + math.ceil((u_i-d)/2)

                #if within boundaries
                if 0 <= v1_test <= 255 and 0 <= v2_test <= 255:
                    bits_remaining = max_len - payload_index
                    bits_to_encode = min(bits_to_hide, bits_remaining)
                    bin_secret = payload[payload_index:payload_index + bits_to_encode]
                    if bits_to_encode < bits_to_hide:
                        bin_secret = bin_secret.ljust(bits_to_hide, '0')
                        
                    secret = int(bin_secret,2)
                    
                    new_d = l_i+secret if d >= 0 else -(l_i+secret)
                    
                    if d%2!=0:
                        v1_prime = v1 - math.ceil((new_d-d)/2)
                        v2_prime = v2 + math.floor((new_d-d)/2)
                    if d%2==0:
                        v1_prime = v1 - math.floor((new_d-d)/2)
                        v2_prime = v2 + math.ceil((new_d-d)/2)
                    payload_index += bits_to_hide
                    
                    array[idx1] = v1_prime
                    array[idx2] = v2_prime
                else:
                    continue  # Skip if pixel values go out of bounds
    # if original was 1‑D, strip off the dummy channel axis
    if array.shape[-1] == 1:
        return array[:, 0]
    return array

def extract(array, ranges=None, range_table=None,seed=None, framing=payloader.FixedHeader, framing_kwargs={}):
    """
    Extracts a payload from a cover array using Value Differencing steganography.

    Args:
        array (numpy.ndarray): Cover object
        ranges (list, optional): pre-defined ranges
        range_table (list, optional): pre-defined  range table
        seed (optional): Seed for PRNG indices
        framing (stegosphere.payloader.PayloadFrame | None): Type of framing used for payload
        framing_kwargs (dict): Specific arguments for payload framing

    Returns:
        str: extracted payload
    """
    if array.ndim == 1:
        array = array[:, None]  # shape (N,) → (N,1)
    
    spatial_dim = array.ndim - 1
    channel_dim = array.shape[-1]

    if range_table is None:
        if ranges is None:
            ranges =  define_range()
        range_table = define_range_table(ranges)

    bin_payload = ""
    value_pairs = _get_pairs(array, spatial_dim, seed)
    for (v1_coords, v2_coords) in value_pairs:
        for channel in range(channel_dim):
            idx1 = v1_coords + (channel,)
            idx2 = v2_coords + (channel,)
            v1 = int(array[idx1])
            v2 = int(array[idx2])
            d = v2 - v1
            abs_d = abs(d)

            try:
                bits_retrieved, l_i, u_i = range_table[abs_d]
            except Exception:
                continue  # Skip if difference is out of range
            if bits_retrieved > 0:
                #Test boundaries
                if d%2!=0:
                    v1_prime = v1 - math.ceil((u_i-d)/2)
                    v2_prime = v2 + math.floor((u_i-d)/2)
                if d%2==0:
                    v1_prime = v1 - math.floor((u_i-d)/2)
                    v2_prime = v2 + math.ceil((u_i-d)/2)

                if 0 <= v1_prime <= 255 and 0 <= v2_prime <= 255:
                    s = abs_d - l_i
                    bits = bin(s)[2:].zfill(bits_retrieved)
                    bin_payload += bits
                else: 
                    continue  # Skip if pixel values go out of bounds
    
    payload = io.decode_binary_payload(bin_payload, framing, framing_kwargs)
    
    return payload



def define_range(range_offset=3, range_start=1, range_range=(0,255)):
    """
    Defines ranges used in Value Differencing.
    Output format is [(bits changed, lower bound,  upper bound),...]
    The bits changed in the lowest range is given by range_start.
    The upper bound in the lowest range is given by range_offset.
    The overall lower and upper bound is given by range_range.
    
    Args:
        range_offset (int, optional): ...
        range_start (int, optional): ...
        range_range (tuple, optional): ...
    """
    min_value, max_value = range_range
    max_value += abs(min_value)
    min_value = 0
    lower = 2**range_offset - 1
    ranges = [(range_start,0,lower)]
    lower += 1
    n = 1 + range_offset
    while lower <= max_value:
        upper = min((2**n)-1,max_value)
        ranges.append((n-range_offset+range_start,lower,upper))
        lower = upper + 1
        n += 1       
    return ranges


def define_range_table(ranges):
    """
    Return a list with fast access of bits to hide and bounds.
    Input can be output of `define_range` function.
    
    Args:
        ranges (list): List of ranges in the format of [(bits to change, lower bound, upper bound), (bits to change + 1, lower bound, upper bound),...]
    """
    table = []
    for bits, lo, hi in ranges:
        for d in range(lo, hi+1):
            table.append([bits, lo, hi])
    return table

#replace with selector PRNG function!
def _get_pairs(array, spatial_dim, seed=None):
    #position pairs
    indices = np.ndindex(*array.shape[:spatial_dim])
    indices = list(indices)
    if seed is not None:
        np.random.default_rng(seed).shuffle(indices)
    pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1,2)]
    return pairs
