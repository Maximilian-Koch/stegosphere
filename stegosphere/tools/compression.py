"""
Helper functions for applying compression.
"""

import lzma
import zlib
import base64

__all__ = ['compress', 'decompress', 'binary_compress', 'binary_decompress']


def bits_to_bytes(binary_string):
    """
    Turns string of bits into bytes object with padding

    Args:
        binary_string (str): String of bits
    Returns:
        tuple: Tuple containing bytes and padded length
    """
    pad_len = (8 - len(binary_string) % 8) % 8
    binary_string += '0' * pad_len
    byte_array = bytearray()
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        byte_array.append(int(byte, 2))
    return bytes(byte_array), pad_len

def bytes_to_bits(byte_data, pad_len):
    """
    Turns bytes into binary string. Reverse function of `bits_to_bytes`.

    Args:
        byte_data (bytes): Bytes
        pad_len (int): Length to pad
    Returns:
        str: String of bits
    """
    bits = ''.join(f'{byte:08b}' for byte in byte_data)
    if pad_len > 0:
        bits = bits[:-pad_len]
    return bits

def compress(string, method='lzma'):
    """
    Compress a string using lzma or deflate (zlib) algorithm.

    Args:
        string (str): The string to be compressed
        method (str): The method to be used. Either `lzma` or `deflate`
    Returns:
        str: the compressed string
    """
    if method=='lzma':
        compressed = lzma.compress(bytes(string, 'utf-8'), preset=9 | lzma.PRESET_EXTREME)
    elif method=='deflate':
        compressed = zlib.compress(bytes(string, 'utf-8'))
    else:
        raise NotImplementedError
    return base64.b64encode(compressed).decode('ascii')

def decompress(string, method='lzma'):
    """
    Decompress a bytes object using lzma or deflate (zlib) algorithm.

    Args:
        string (str): The compressed string
        method (str): The method to be used. Either `lzma` or `deflate`
    Returns:
        str: The decompressed string.
    """
    string = base64.b64decode(string)
    if method=='lzma':
        return lzma.decompress(string)
    elif method=='deflate':
        return zlib.decompress(string)
    
def binary_compress(binary_string, method='lzma'):
    """
    Compresses a binary string using lzma and returns the compressed binary string.
    The padding length is stored in the first 8 bits.

    Args:
        binary_string (str): String of bits
        method (str): Compression method. Only `lzma` available.
    Returns:
        str: Compressed binary string.
    Raises:
        NotImplementedError: if other method is used
    """
    if method == 'lzma':
        byte_data, pad_len = bits_to_bytes(binary_string)
        compressed_data = lzma.compress(byte_data, preset=9 | lzma.PRESET_EXTREME)
        compressed_bits = ''.join(f'{byte:08b}' for byte in compressed_data)
        pad_len_bits = f'{pad_len:08b}'
        return pad_len_bits + compressed_bits
    else:
        raise NotImplementedError

def binary_decompress(compressed_binary_string, method='lzma'):
    """
    Decompresses a compressed binary string back to the original binary string.

    Args:
        compressed_binary_string (str): Compressed string
        method (str): Compression method. Only `lzma` available.
    Returns:
        str: Decompressed binary string.
    Raises:
        NotImplementedError: if other method is used 
    """
    if method == 'lzma':
        pad_len_bits = compressed_binary_string[:8]
        pad_len = int(pad_len_bits, 2)
        compressed_bits = compressed_binary_string[8:]
        compressed_bytes = bytearray()
        for i in range(0, len(compressed_bits), 8):
            byte = compressed_bits[i:i+8]
            compressed_bytes.append(int(byte, 2))
        compressed_bytes = bytes(compressed_bytes)
        byte_data = lzma.decompress(compressed_bytes)
        original_bits = bytes_to_bits(byte_data, pad_len)
        return original_bits
    else:
        raise NotImplementedError
