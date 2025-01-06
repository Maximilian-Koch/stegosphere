import re

from stegosphere.tools import compression

def encode_payload(payload, method='metadata', metadata_length=32, delimiter_message='###END###', compress=False):
    """
    Prepares a payload for embedding.

    :param payload: The payload to be embedded
    :type payload: str
    :param method: The method for end-of-message signifying, either 'metadata', 'delimiter' or None
    :type method: str/None
    :param metadata_length: The length of the block denoting the number of embedded bits.
    :type metadata_length: int
    :param delimiter_message: The delimiter to use as end-of-message
    :type delimiter_message: str
    :param compress: Use compression to compress the payload
    :type compress: bool, str
    """
    payload_format = check_type(payload)
    if payload_format == 1:
        #Hex messages are a common output after encryption, thus treated specifically
        payload = hex_to_binary(payload)
    elif payload_format == 2:
        payload = data_to_binary(payload)

    if compress:
        previous_length = len(payload)
        payload = compression.binary_compress(payload, compress)
        after_length = len(payload)
        if after_length > previous_length:
            warnings.warn('message length increased due to compression. That might be the case for already compressed data.')
        
    if method == 'delimiter':
        if not is_binary(delimiter_message):
            delimiter_message = data_to_binary(delimiter_message)
        payload += delimiter_message
        assert payload.count(delimiter_message)==1, Exception('Delimiter appears in data. Use another delimiter or change data minimally.')
    elif method == 'metadata':
        payload = f"{len(payload):0{metadata_length}b}" + payload
    elif method is not None:
        raise Exception('Method must be either delimiter, metadata or None.')
    
    return payload


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

