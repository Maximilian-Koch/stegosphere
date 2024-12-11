import numpy as np

import os
import re
import ctypes
import warnings

DELIMITER_MESSAGE = '###END###'
#32 bit for metadata length should be sufficient for almost all usecases
#Set higher if hidden message exceeds ~0.5GB
#Set lower if hidden message needs to be minimized
METADATA_LENGTH_IMAGE = 32
METADATA_LENGTH_AUDIO = 32
METADATA_LENGTH_VIDEO = 32
METADATA_LENGTH_LSB = 32

C_AVAILABLE = True
try:
    backend = ctypes.CDLL('./c_steg.so')
    backend.parse_binary_message.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    backend.generate_message_bits.argtypes = [ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    backend.generate_message_bits.restype = None
except:
    warnings.warn('C backend could not be loaded. Methods will use slower Python.')
    C_AVAILABLE = False
    
class File:
    """
    A superclass for handling file data in various formats.
    Provides common functionality for reading, saving, and reshaping data.

    :param data: The file data, either a NumPy array, file path, or list/tuple.
    :type data: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, data):
        self._shape = None
        self.data = self.read(data)
        if hasattr(self.data, 'shape'):
            self._shape = self.data.shape

    def read(self, data):
        """
        Reads and returns data from various sources such as NumPy arrays, file paths, or lists/tuples.

        :param data: Data source, either a file path, NumPy array, or list/tuple.
        :type data: str, numpy.ndarray, list, or tuple
        :return: Data as a NumPy array.
        :rtype: numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            return data
        elif os.path.isfile(data):
            return self._read_file(data)  # Specific file reading logic in subclass
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            raise Exception(f'{self.__class__.__name__} format not supported')

    def flush(self):
        """
        Reshapes the file data back to its original shape.
        Use if self.save is not sufficient for saving the result.
        """
        if self._shape is not None:
            self.data = self.data.reshape(self._shape)
        self._flush_file()

    def save(self, path):
        """
        Saves the current data to the specified file path.

        :param path: File path where the data will be saved.
        :type path: str
        """
        self.flush()
        self._save_file(path)  # Specific file saving logic in subclass

    def _flush_file(self):
        """
        Abstract method to be implemented in subclasses for flushing specific file types.
        """
        pass

    def _read_file(self, path):
        """
        Abstract method to be implemented in subclasses for reading specific file types (e.g., image, audio).

        :param path: The file path to read from.
        :type path: str
        """
        raise NotImplementedError("Subclasses must implement `_read_file` method")

    def _save_file(self, path):
        """
        Abstract method to be implemented in subclasses for saving specific file types (e.g., image, audio).

        :param path: The file path to save to.
        :type path: str
        """
        raise NotImplementedError("Subclasses must implement `_save_file` method")
    

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
    

