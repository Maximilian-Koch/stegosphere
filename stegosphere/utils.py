import numpy as np

import os
import re
import ctypes
import warnings

import compression

DELIMITER_MESSAGE = '###END###'
#32 bit for metadata length should be sufficient for almost all usecases
#Set higher if hidden message exceeds ~0.5GB
#Set lower if hidden message needs to be minimized
METADATA_LENGTH_IMAGE = 32
METADATA_LENGTH_AUDIO = 32
METADATA_LENGTH_VIDEO = 32
METADATA_LENGTH_LSB = 32

ANALYSIS = True

C_AVAILABLE = True
so_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "c_steg.so")
try:
    backend = ctypes.CDLL(so_file_path)
    backend.parse_binary_message.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    backend.generate_message_bits.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    backend.generate_message_bits.restype = None
except:
    warnings.warn('C backend could not be loaded. Methods will use slower Python.')
    C_AVAILABLE = False


def encode_message(message, method, metadata_length, delimiter_message, compress):
    message_format = check_type(message)
    if message_format == 1:
        #Hex messages are a common output after encryption, thus treated specifically
        message = hex_to_binary(message)
    elif message_format == 2:
        message = data_to_binary(message)

    if compress:
        previous_length = len(message)
        message = compression.binary_compress(message, compress)
        after_length = len(message)
        if after_length > previous_length:
            warnings.warn('message length increased due to compression. That might be the case for already compressed data.')
        
    if method == 'delimiter':
        if not is_binary(delimiter_message):
            delimiter_message = data_to_binary(delimiter_message)
        message += delimiter_message
    elif method == 'metadata':
        message = f"{len(message):0{metadata_length}b}" + message
    elif method is not None:
        raise Exception('Method must be either delimiter, metadata or None.')
    
    if method == 'delimiter':
        assert message.count(delimiter_message)==1, Exception('Delimiter appears in data. Use another delimiter or change data minimally.')
    
    return message
    

def file_to_binary(path):
    """
    Converts a file into its binary representation.
    
    :param file_path: Path to the input file
    :return: Binary data of the file
    """
    with open(path, 'rb') as file:
        binary_data = file.read()
    return data_to_binary(binary_data)

def binary_to_file(binary_data, output_path):
    """
    Converts binary data back to a file.
    
    :param binary_data: Binary data to convert
    :param output_file_path: Path to save the output file
    """
    data = binary_to_data(binary_data)
    with open(output_path, 'wb') as file:
        file.write(data)

def data_to_binary(data):
    """
    Converts data (string or bytes) to a binary string.
    
    :param data: Data to convert
    :return: Binary string
    """
    if isinstance(data, str):
        return ''.join(format(ord(char), '08b') for char in data)
    elif isinstance(data, bytes) or isinstance(data, np.ndarray):
        return ''.join(format(byte, '08b') for byte in data)
    elif isinstance(data, int) or isinstance(data, np.uint8):
        return format(data, '08b')
    else:
        raise TypeError("Type not supported.")

def binary_to_data(binary):
    """
    Converts binary string to data.
    
    :param binary: Binary string to convert
    :return: Data as bytes
    """
    all_bytes = [binary[i: i + 8] for i in range(0, len(binary), 8)]
    return bytes([int(byte, 2) for byte in all_bytes])

def is_binary(data):
    """
    Check if string only contains binary data.

    :param data: data to be checked
    :return: bool
    """
    return re.fullmatch(r'[01]+', data)


def hex_to_binary(data):
    """
    Convert hexadecimal str to binary str.

    :param data: data to be converted
    :return: binary data
    """
    return bin(int(data, 16))[2:]

def check_type(data):
    """
    Check whether data is binary, hexadecimal or neither

    :param data: data to be checked
    :return: int, 0 meaning binary, 1 meaning hex, 2 meaning neither
    """
    if is_binary(data):
        return 0 #binary
    elif re.fullmatch(r'[0-9a-fA-F]+', data):
        return 1 #hexadecimal
    else:
        return 2 #to be treated as string


def dtype_range(dtype):
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
    else:
        raise Exception('array dtype must be integer or float.')
    return info.min, info.max

def prng_indices(length, key):
    if type(key)!=int:
        key = np.frombuffer(key.encode(), dtype=np.uint32)
    rng = np.random.default_rng(seed=key)
    indices = np.arange(length)
    rng.shuffle(indices)
    return indices

def parse_message(message, bits):
    if C_AVAILABLE:
        message_len = len(message)
        output_len = message_len // bits
        output_array = (ctypes.c_int * output_len)()
        backend.parse_binary_message(message.encode('ascii'), bits, output_array, message_len)
        return np.array(output_array[:output_len])
    else:
        return np.array([int(message[i:i + bits], 2) for i in range(0, len(message), bits)])
        

def generate_message_bits(extracted_bits, bits):
    if C_AVAILABLE:
        length = len(extracted_bits)
        extracted_bits = np.ascontiguousarray(extracted_bits, dtype=np.int32)
        output_buffer = ctypes.create_string_buffer(length*bits+1)
        backend.generate_message_bits(extracted_bits.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int(length),ctypes.c_int(bits),output_buffer)
        return output_buffer.value.decode('ascii')
    else:
        return ''.join(f'{bit:0{bits}b}' for bit in extracted_bits)


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
    
