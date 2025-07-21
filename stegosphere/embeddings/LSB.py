"""
This implements Least Significant Bit steganography.
"""

import warnings
import ctypes
import math
import os

import numpy as np

from stegosphere import io, payloader
from stegosphere.selector import prng_indices

BACKEND = True
try:
    so_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend", "lsb.dll"))
    backend = ctypes.CDLL(so_file_path)
    backend.embed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    backend.embed.restype = None
    backend.extract.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    backend.extract.restype = None
    
except (OSError, AttributeError) as e:
    warnings.warn(f'C backend could not be loaded ({type(e).__name__}): {e}. Falling back to Python.')
    BACKEND = False


def max_capacity(array, bits = 1):
    """
    Calculates the maximum capacity of the cover object for embedding a message.
    The capacity is determined by the size of the data array and the number of bits available for modification.
    Args:
        array (numpy.ndarray): Cover object
        bits (int, optional): Number of bits changed per value. Defaults to 1. 
    Returns:
        int: Maximum capacity of the object in bits
    """
    return array.size * bits


def embed(array, payload, matching=False, seed=None, indices=None, bits=1,
          framing=payloader.FixedHeader, framing_kwargs={}):
    """
    Encodes a message into the cover data using LSB steganography,
    with optional LSB-matching, optional PRNG or custom indices,
    and both Python and C backends.

    Args:
        array (numpy.ndarray): Cover object
        payload (str):  String of binary numbers
        matching (bool, optional): Whether to use LSB matching (`True`) or LSB replacement (`False`). Only available with Python backend and `bits=1`.
        seed (optional): Seed for PRNG indices
        indices (optional): Indices to embed payload into. Can't use `seed` and `indices` at the same time. Use PRNG selector strategies instead.
        bits (int, optional): Number of bits to change per value.
        framing (stegosphere.payloader.PayloadFrame | None): Type of framing used for payload
        framing_kwargs (dict): Specific arguments for payload framing
    Returns:
        numpy.ndarray: Array with embedded payload
    Raises:
        ValueError: Not enough indices for embedding.
        NotImplementedError: When using `matching=True` with more than one `bits`
    Warns:
        UserWarning: If array is too small, the payload will be shortened
    """
    # 1. Frame & encode payload to bitstring
    bitstr = io.encode_binary_payload(payload, framing, framing_kwargs)
    total_capacity = (array.size * bits) if indices is None else indices.size*bits
    if len(bitstr) > total_capacity:
        if indices is not None:
            raise ValueError(f"indices too short for payload: need {len(bitstr)} bits, got capacity {total_capacity}")
        warnings.warn("Insufficient capacity: payload will be truncated to fit.", UserWarning)
        bitstr = bitstr[:total_capacity]

    orig_shape = array.shape

    # 2. Python fallback path
    if not globals().get('BACKEND'):
        # flatten and work in int32
        content = array.copy().ravel().astype(np.int32)
        mask = (1 << bits) - 1

        # split into chunks of size `bits`
        payload_array = np.array(
            [int(bitstr[i:i+bits], 2)
             for i in range(0, len(bitstr), bits)],
            dtype=np.int32
        )
        L = len(payload_array)

        # build indices array
        if seed is not None:
            idxs = prng_indices(content.size, seed).astype(np.int32)
        elif indices is None:
            idxs = np.arange(content.size, dtype=np.int32)
        else:
            idxs = np.asarray(indices, np.int32)

        if idxs.shape[0] < L:
            raise ValueError(f"indices too short: need {L} entries, got {idxs.shape[0]}")
        idxs = idxs[:L]

        # (a) LSB-matching for bits=1
        if matching:
            if bits != 1:
                raise NotImplementedError("LSB matching only supported for bits=1 in this implementation")
            vals = content[idxs]
            mism = (vals & 1) != payload_array
            M = int(mism.sum())
            if M > 0:
                # random +/-1 per mismatch
                dirs = np.where(np.random.rand(M) < 0.5, -1, +1)
                bad = idxs[mism]
                content[bad] = vals[mism] + dirs
            # enforce bounds
            info = np.iinfo(array.dtype)
            content = np.clip(content, info.min, info.max)
        # (b) simple LSB replacement
        else:
            content[idxs] &= ~mask
            content[idxs] |= payload_array

        # reshape & cast back
        return content.reshape(orig_shape).astype(array.dtype)

    # 3. C backend path
    content = array.copy().ravel()
    bitstr_bytes = bitstr.encode('utf-8')
    value_size = content.dtype.itemsize
    ptr = content.ctypes.data_as(ctypes.c_void_p)

    # build & cast indices for C
    if seed is not None:
        idx = prng_indices(content.size, seed)[:len(bitstr_bytes)].astype(np.int32)
        idx_ptr = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    elif indices is not None:
        idx = np.asarray(indices, np.int32)
        idx_ptr = idx.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        idx_ptr = None

    #!! SUMMARISE CODE with python backend

    # C embed ignores `matching` flag; matching only in Python path
    backend.embed(ptr, content.size, bitstr_bytes, bits, value_size, idx_ptr)

    return content.reshape(orig_shape).astype(array.dtype)


def extract(array, seed=None, indices=None, bits=1, adaptive=False,
        framing=payloader.FixedHeader, framing_kwargs={}):
    """
    Decodes a message from the cover data using LSB steganography. Works with LSB  matching and replacement.

    Args:
        array (numpy.ndarray): Cover object
        seed (optional): Seed for PRNG indices
        indices (optional): Indices to embed payload into. Can't use `seed` and `indices` at the same time. Use PRNG selector strategies instead.
        bits (int, optional): Number of bits to extract per value.
        framing (stegosphere.payloader.PayloadFrame | None): Type of framing used for payload
        framing_kwargs (dict): Specific arguments for payload framing
    Returns:
        str: extracted payload
    """
    content = array.copy().flatten()
    
    if seed is not None:
        indices = prng_indices(content.size, seed).astype(np.int32)
        indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    elif indices is not None:
        indices = np.asarray(indices, np.int32)
        indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        indices = None
        indices_pointer = None


    def extract_bits_python(values, bits, indices_pointer=None):
        mask = (1 << bits) - 1
        return ''.join(f'{(v & mask):0{bits}b}' for v in values)

    def extract_bits_c(values, bits, indices_pointer):
        values_flat = values.ravel()
        if not np.issubdtype(values_flat.dtype, np.integer):
            raise ValueError("Array must have an integer dtype for LSB extraction.")
        
        values_cast = values_flat.astype(np.int64, copy=False)

        ptr = values_cast.ctypes.data_as(ctypes.c_void_p)
        length = values_cast.size
        elem_size = values_cast.dtype.itemsize

        total_bits = bits * length
        out_buffer = ctypes.create_string_buffer(total_bits + 1)  # +1 for null terminator
        
        backend.extract(ptr, length, bits, elem_size, out_buffer, indices_pointer)
        
        return out_buffer.value.decode('utf-8')

    #not currently used, but will play a role in adaptive extracting
    if BACKEND is True:
        backend_extract = extract_bits_c
    else:
        backend_extract = extract_bits_python

 
    message_bits = ""

    if adaptive is False:
        total_bits = len(content) * bits

        if globals().get('BACKEND'):
            if indices is not None:
                slice_ = content[indices]
                raw_bits = extract_bits_c(slice_, bits, None)
            else:
                raw_bits = extract_bits_c(content, bits, None)
        else:
            needed = math.ceil(total_bits / bits)
            subset = content[ indices[:needed] if indices is not None else np.arange(needed) ]
            raw_bits = extract_bits_python(subset, bits)

        message_bits = io.decode_binary_payload(raw_bits[:total_bits], framing, framing_kwargs)
    return message_bits

#earlier versions of an adaptive version
"""
        n_bits = len(content) * bits
        
        needed_elems = math.ceil(n_bits / bits)
        subset = content[indices[:needed_elems]]
        
        raw_bits = backend_extract(subset, bits, indices_pointer)
        message_bits = raw_bits[:n_bits]
        message_bits = io.decode_binary_payload(message_bits, framing, framing_kwargs)
    return message_bits

    ADAPTIVE PROTOTYPE
    elif isinstance(framing, payloader.FixedHeader):
        total_metadata_pixels = math.ceil(metadata_length / bits)
        metadata_subset = content[indices[:total_metadata_pixels]]
        
        metadata_raw = backend_extract(metadata_subset, bits, indices_pointer)
        metadata_raw = metadata_raw[:metadata_length]
        message_length = int(metadata_raw, 2)
        
        total_message_pixels = math.ceil(message_length / bits)
        message_subset = content[indices[total_metadata_pixels : total_metadata_pixels + total_message_pixels]]
        
        message_raw = backend_extract(message_subset, bits, indices_pointer)
        message_bits = message_raw[:message_length]

    elif isinstance(framing, payloader.Delimiter):
        if not io.is_binary(delimiter_message):
            delimiter = io.data_to_binary(delimiter_message)
        else:
            delimiter = delimiter_message
    
        mask = (1 << bits) - 1

        #not efficient for large arrays. replace with reading given amount of values before checking for delimiter
        for idx in indices:
            pixel_val = content[idx]
            extracted_bits = f'{(pixel_val & mask):0{bits}b}'     
            message_bits += extracted_bits
            if message_bits.endswith(delimiter):
                message_bits = message_bits[:-len(delimiter)]  # remove delimiter
                break
    else:
        raise ValueError(f"Invalid method: {method}")

    return message_bits
    """
