"""
Utilities for converting between file, string, and binary formats, and
for encoding/decoding payloads with optional framing.

Functions include:
- Encoding/decoding binary payloads with framing
- Conversion between files, strings, bytes, hex, and binary
- Type checks for binary and hexadecimal strings
"""

import re
import numpy as np

from stegosphere.tools import compression
from stegosphere import payloader

__all__ = ['encode_binary_payload','decode_binary_payload',
    'file_to_binary','binary_to_file',
    'data_to_binary','binary_to_data',
    'is_binary','hex_to_binary','check_type']

def encode_binary_payload(payload, framing=None, framing_kwargs=None):
    """
    Converts payload into binary and applies payload framing.

    Args:
        payload (str): Payload to be encoded
        framing (stegosphere.payloader.PayloadFrame | None, optional): Framing method to use
        framing_kwargs (dict | None): specific framing arguments
    Returns:
        str: Encoded binary payload
    """
    framing_kwargs = framing_kwargs or {}
    payload_format = check_type(payload)
    if payload_format == 1:
        #Hex messages are a common output after encryption, thus treated specifically
        payload = hex_to_binary(payload)
    elif payload_format == 2:
        payload = data_to_binary(payload)
    payload = payloader.encode_framing(payload, framing, framing_kwargs)
    return payload

def decode_binary_payload(payload, framing, framing_kwargs=None):
    """
    Decodes binary payload by removing payload framing.

    Args:
        payload (str): Payload to be decoded
        framing (stegosphere.payloader.PayloadFrame | None, optional): Framing method to use
        framing_kwargs (dict | None): specific framing arguments
    Returns:
        str: Decoded binary payload
    """
    framing_kwargs = framing_kwargs or {}
    payload = payloader.decode_framing(payload, framing, framing_kwargs)
    return payload

def file_to_binary(path):
    """
    Converts a file into its binary representation.

    Args:
        path (str): Path to the input file

    Returns:
        str: Binary data of the file
    """
    with open(path, 'rb') as file:
        binary_data = file.read()
    return data_to_binary(binary_data)

def binary_to_file(binary_data, output_path):
    """
    Write binary data into a file.
    
    Args:
        binary_data (str): Binary data to store
        output_path (str): Path to save the output file
    
    Returns:
        None
    """
    data = binary_to_data(binary_data)
    with open(output_path, 'wb') as file:
        file.write(data)

def data_to_binary(data):
    """
    Converts data (string or bytes) to a binary string.
    
    Args:
        data: Data to convert
    
    Returns:
        str: Binary string
    
    Raises:
        TypeError: If `data` could not be converted
    """
    if isinstance(data, str):
        return ''.join(format(ord(char), '08b') for char in data)
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return ''.join(format(byte, '08b') for byte in data)
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, '08b')
    raise TypeError(f"Cannot convert type '{type(data).__name__}' to binary")

def binary_to_data(binary):
    """
    Converts binary string to data.

    Args:
        binary (str): Binary string to convert
    
    Returns:
        bytes: Data as bytes
    """
    all_bytes = [binary[i: i + 8] for i in range(0, len(binary), 8)]
    return bytes([int(byte, 2) for byte in all_bytes])

def is_binary(data):
    """
    Check if string only contains binary data.

    Args:
        data (str): data to be checked
    
    Returns:
        bool: True, if `data` is binary. False, if not.
    """
    return bool(re.fullmatch(r'[01]+', data))


def hex_to_binary(data):
    """
    Convert hexadecimal str to binary str.

    Args:
        data (str): data to be converted
    Returns:
        str: Binary data
    """
    return bin(int(data, 16))[2:]

def check_type(data):
    """
    Check whether data is binary, hexadecimal or neither.

    Args:
        data (str): data to be checked
    
    Returns:
        int: 0 meaning binary, 1 meaning hex, 2 meaning neither
    """
    if is_binary(data):
        return 0 #binary
    elif re.fullmatch(r'[0-9a-fA-F]+', data):
        return 1 #hexadecimal
    else:
        return 2 #to be treated as string