import lzma

__all__ = ['compress', 'decompress']

def bits_to_bytes(binary_string):
    pad_len = (8 - len(binary_string) % 8) % 8
    binary_string += '0' * pad_len
    byte_array = bytearray()
    for i in range(0, len(binary_string), 8):
        byte = binary_string[i:i+8]
        byte_array.append(int(byte, 2))
    return bytes(byte_array), pad_len

def bytes_to_bits(byte_data, pad_len):
    bits = ''.join(f'{byte:08b}' for byte in byte_data)
    if pad_len > 0:
        bits = bits[:-pad_len]
    return bits

def compress(binary_string):
    """
    Compresses a binary string using lzma and returns the compressed binary string.
    The padding length is stored in the first 8 bits.
    """
    byte_data, pad_len = bits_to_bytes(binary_string)
    compressed_data = lzma.compress(byte_data, preset=9 | lzma.PRESET_EXTREME)
    compressed_bits = ''.join(f'{byte:08b}' for byte in compressed_data)
    pad_len_bits = f'{pad_len:08b}'
    return pad_len_bits + compressed_bits

def decompress(compressed_binary_string):
    """
    Decompresses the compressed binary string back to the original binary string.
    """
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

