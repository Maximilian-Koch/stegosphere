import warnings
import math

import numpy as np


from stegosphere import base
from stegosphere import utils
from stegosphere import io
from stegosphere.config import METADATA_LENGTH_VD, DELIMITER_MESSAGE

"""
This is an adapted and generalised version of the Pixel Value Differencing method as proposed by
Wu, D. C., & Tsai, W. H. (2003).
A steganographic method for images by pixel-value differencing.
Pattern recognition letters, 24(9-10), 1613-1626.

"""

def embed(array, payload, spatial_dim=None, channel_dim=None,
          ranges=None, range_offset=3, range_start=1,
          range_range=None,
          seed=None, method='metadata', metadata_length=METADATA_LENGTH_VD,
          delimiter_message=DELIMITER_MESSAGE, compress=False):
            
    """
    Embed a payload into a cover array using Value Differencing steganography.


    :param payload: The payload to be hidden. Gets converted into binary if not already.
    :type payload: str
    :param spatial_dim: Number of spatial dimensions of the data (usually 2 in images, 1 in audio, 3 in video)
    :type spatial_dim: int
    :param channel_dim: Number of channels of the data (3 in RGB, 2 in stereo audio, ...)
    :type channel_dim: int
    :param ranges: pre-defined ranges.
    :type ranges: list
    :param range_offset: ...
    :type range_offset: ...
    :param range_start: ...
    :type range_start: ...
    :param range_range: Minimum and maximum a value can take in the array.
    :type range_range: tuple
    :param seed: (Optional) Seed value for pseudo-randomly distributing the message in the cover data.
    :type seed: int, optional
    :param method: Method for marking the end of the message. Options are 'delimiter', 'metadata', or None. Defaults to 'metadata'.
    :type method: str, optional
    :param metadata_length: Length of the metadata in bits when `method='metadata'`. Defaults to `METADATA_LENGTH_IMAGE`.
    :type metadata_length: int, optional
    :param delimiter_message: The delimiter string used when `method='delimiter'`.
    :type delimiter_message: str, optional

    :return: True, if it worked.
    :rtype: bool
    """
    array = array.copy()
    
    if spatial_dim is None or channel_dim is None:
        if len(array.shape) == 1:
            spatial_dim = 1
            channel_dim = 0
        else:
            spatial_dim = len(array.shape) - 1
            channel_dim = 1
    if ranges is None:
        if range_range is None:
            range_range = utils.dtype_range(array.dtype)
        ranges = _define_range(range_offset, range_start, range_range)

    payload = io.encode_payload(payload, method, metadata_length, delimiter_message, compress)
    value_pairs = _get_pairs(array, spatial_dim, seed)
    payload_index = 0
    max_len = len(payload)
    
    for (v1_coords, v2_coords) in value_pairs:
        if payload_index >= max_len:
            break
        for channel in range(channel_dim):
            v1 = int(array[*v1_coords,channel])
            v2 = int(array[*v2_coords,channel])
            d = v2-v1
            abs_d = abs(d)

            try:
                bits_to_hide, l_i, u_i = _range(ranges, abs_d)
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
                    
                    array[*v1_coords, channel] = v1_prime
                    array[*v2_coords, channel] = v2_prime
                else:
                    
                    continue  # Skip if pixel values go out of bounds

    return array

def extract(array, spatial_dim=None, channel_dim=None,
           ranges=None, range_offset=3, range_start=1, range_range=None,
           seed=None, method='metadata', n_bits=100, metadata_length=METADATA_LENGTH_VD,
           delimiter_message=DELIMITER_MESSAGE, compress=False):
    """
    Decodes a message from the cover data using Value Differencing steganography.

    The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
    supports optional message verification to ensure the decoded message matches the original encoded message.

    :param seed: (Optional) Seed value for pseudo-randomly distributing the message in the cover data.
    :type seed: int, optional
    :param method: Method for marking the end of the message. Options are 'delimiter', 'metadata', or None. Defaults to 'metadata'.
    :type method: str, optional
    :param n_bits: Bits to be read out, used when method=None.
    :type n_bits: int, optional
    :param metadata_length: Length of the metadata in bits when `method='metadata'`. Defaults to `METADATA_LENGTH_LSB`.
    :type metadata_length: int, optional
    :param delimiter_message: The delimiter string used when `method='delimiter'`.
    :type delimiter_message: str, optional

    :return: The decoded message
    :rtype: str or tuple
    """
    assert method in ['metadata','delimiter',None], 'Method must be either delimiter, metadata, or None.'


    if spatial_dim is None or channel_dim is None:
        if len(array.shape) == 1:
            spatial_dim = 1
            channel_dim = 0
        else:
            spatial_dim = len(array.shape) - 1
            channel_dim = 1
    if ranges is None:
        if range_range is None:
            range_range = utils.dtype_range(array.dtype)
        ranges = _define_range(range_offset, range_start, range_range)

        
    bin_payload = ""
    value_pairs = _get_pairs(array, spatial_dim, seed)
    for (v1_coords, v2_coords) in value_pairs:
        for channel in range(channel_dim):
            v1 = int(array[*v1_coords,channel])
            v2 = int(array[*v2_coords,channel])
            d = v2 - v1
            abs_d = abs(d)

            try:
                bits_retrieved, l_i, u_i = _range(ranges, abs_d)
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
    
    if method == 'metadata':
        length = int(bin_payload[:metadata_length], 2)
        payload_end = metadata_length + length
        payload = bin_payload[metadata_length:payload_end]
    elif method == 'delimiter':
        if not is_binary(delimiter_message):
            delimiter_message = data_to_binary(delimiter_message)
        payload = bin_payload.split(delimiter_message)[0]
    elif method is None:
        if n_bits is None: 
            return bin_payload
        return bin_payload[:n_bits]

    if compress:
        payload = compression.binary_decompress(payload, compress)
    
    return payload



def _define_range(range_offset, range_start, range_range):
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

def _range(ranges, diff):
    """Return the number of bits to embed and the lower and upper bounds based on the pixel difference."""
    #faster then dictionary access
    for i, l_i, u_i in ranges:
        if l_i<=diff<=u_i:
            return i, l_i, u_i
        
def _get_pairs(array, spatial_dim, seed=None):
    #position pairs
    indices = np.ndindex(*array.shape[:spatial_dim])
    indices = list(indices)
    if seed is not None:
        np.random.default_rng(seed).shuffle(indices)
    pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1,2)]
    return pairs
