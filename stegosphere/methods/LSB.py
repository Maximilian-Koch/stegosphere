import warnings
import ctypes
import math
import os

import numpy as np

from stegosphere.config import METADATA_LENGTH_LSB, DELIMITER_MESSAGE
from stegosphere import io
from stegosphere.utils import prng_indices

BACKEND = True
try:
    so_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "backend\\lsb.so")
    backend = ctypes.CDLL(so_file_path)
    backend.embed.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p,
                              ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int)]
    backend.embed.restype = None
    backend.extract.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
    backend.extract.restype = None
    
except Exception as e:
    print(e)
    warnings.warn('C backend could not be loaded. Methods will use slower Python.')
    BACKEND = False


def max_capacity(array, bits = 1):
    """
    Calculates the maximum capacity of the cover object for embedding a message.

    The capacity is determined by the size of the data array and the number of bits available for modification.

    :param bits: Number of bits changed per value. Defaults to 1.
    :type bits: int
    :return: The maximum capacity of the object in bits.
    :rtype: int
    """
    return array.size * bits


def embed(array, payload, matching=False, seed=None, bits=1,
               method='metadata', metadata_length=METADATA_LENGTH_LSB,
               delimiter_message=DELIMITER_MESSAGE, compress=False):
    """
    Encodes a message into the cover data using LSB steganography.

    The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
    supports optional message verification to ensure the decoded message matches the original encoded message.
    :param array: The array to write the payload into.
    :type array: np.ndarray
    :param payload: The payload to be hidden. Gets converted into binary if not already.
    :type payload: str
    :param matching: Whether to use LSB matching (not implemented yet). Defaults to False.
    :type matching: bool, optional
    :param seed: (Optional) Seed value for pseudo-randomly distributing the message in the cover data.
    :type seed: int, optional
    :param bits: Number of bits used for decoding per value. Defaults to 1.
    :type bits: int, optional
    :param method: Method for marking the end of the message. Options are 'delimiter', 'metadata', or None. Defaults to 'metadata'.
    :type method: str, optional
    :param metadata_length: Length of the metadata in bits when `method='metadata'`. Defaults to `METADATA_LENGTH_LSB`.
    :type metadata_length: int, optional
    :param delimiter_message: The delimiter string used when `method='delimiter'`.
    :type delimiter_message: str, optional
    :param compress: Whether to use compression on the input data. Defaults to False.
    :type compress: bool, optional

    :raises NotImplementedError: If LSB matching is used.

    :return: True, if it worked.
    :rtype: bool
    """
    if matching is not False: raise NotImplementedError("LSB matching not implemented in current version.")
    
    payload = io.encode_payload(payload, method, metadata_length, delimiter_message, compress)
    
    if len(payload) > max_capacity(array, bits):
        warnings.warn("Insufficient bits, need larger cover or smaller message.")

    orig_dim = array.shape

    if BACKEND is True:
        content = array.flatten().copy()
        payload = payload.encode('utf-8')
        value_size = content.dtype.itemsize
        array_pointer = content.ctypes.data_as(ctypes.c_void_p)

        if seed is not None:
            indices = prng_indices(content.size, seed)
            indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        else:
            indices_pointer = None
        
        backend.embed(array_pointer, content.size, payload, bits, value_size, indices_pointer)

        
    else:
        content = array.copy().flatten().astype(np.int32)
        mask = (1<<bits)-1
        payload_array = np.array([int(payload[i:i + bits], 2) for i in range(0, len(payload), bits)])
        
        payload_length = len(payload_array)
        if seed:
            indices = prng_indices(len(content),seed)[:payload_length]
        else:
            indices = np.arange(payload_length)
        
        content[indices[:payload_length]] &= ~mask
        content[indices[:payload_length]] |= payload_array

    content = content.reshape(orig_dim).astype(array.dtype)

    return content


def extract(array, matching=False, seed=None, bits=1, method='metadata', n_bits=100, 
            metadata_length=METADATA_LENGTH_LSB, delimiter_message=DELIMITER_MESSAGE,
            compress=False):
    """
    Decodes a message from the cover data using LSB steganography.

    :param matching: Whether to use LSB matching (not implemented yet).
    :param seed: Seed value for pseudo-randomly distributing the message in the cover data.
    :param bits: Number of bits used for decoding per value.
    :param method: Method for marking the end of the message: 'delimiter', 'metadata', or None.
    :param n_bits: Bits to be read if method=None.
    :param metadata_length: Length of the metadata in bits if method='metadata'.
    :param delimiter_message: Delimiter used if method='delimiter'.
    :param compress: Whether compression was used on the encoded data.
    :return: The decoded message bits (or a final string if you decompress).
    """
    if matching:
        raise NotImplementedError("LSB matching not implemented in current version.")
    assert method in ['metadata','delimiter', None]
    content = array.copy().flatten()
    
    if seed is not None:
        indices = prng_indices(len(content), seed)
        indices_pointer = indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
    else:
        indices = np.arange(len(content))
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

    if BACKEND is True:
        backend_extract = extract_bits_c
    else:
        backend_extract = extract_bits_python


    message_bits = ""

    if method is None:
        if n_bits is None:
            n_bits = len(content) * bits
        
        needed_elems = math.ceil(n_bits / bits)
        subset = content[indices[:needed_elems]]
        
        raw_bits = backend_extract(subset, bits, indices_pointer)
        message_bits = raw_bits[:n_bits]

    elif method == 'metadata':
        total_metadata_pixels = math.ceil(metadata_length / bits)
        metadata_subset = content[indices[:total_metadata_pixels]]
        
        metadata_raw = backend_extract(metadata_subset, bits, indices_pointer)
        metadata_raw = metadata_raw[:metadata_length]
        message_length = int(metadata_raw, 2)
        
        total_message_pixels = math.ceil(message_length / bits)
        message_subset = content[indices[total_metadata_pixels : total_metadata_pixels + total_message_pixels]]
        
        message_raw = backend_extract(message_subset, bits, indices_pointer)
        message_bits = message_raw[:message_length]

    elif method == 'delimiter':
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

    if compress:
        message_bits = compression.binary_decompress(message_bits, compress)

    return message_bits
