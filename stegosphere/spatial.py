import numpy as np
import math
import itertools
import os
import warnings

import utils
import compression
from utils import *

__all__ = ['BaseLSB','BaseVD','CrossVD']

class BaseCoder:
    def __init__(self):
        pass
    def _encode_message(self, message, method, metadata_length, delimiter_message, compress):
        message_format = utils.check_type(message)
        if message_format == 1:
            #Hex messages are a common output after encryption, thus treated specifically
            message = hex_to_binary(message)
        elif message_format == 2:
            message = data_to_binary(message)

        if compress:
            previous_length = len(message)
            message = compression.compress(message)
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
    def _decode_message(self, seed, method, metadata_length, delimiter_message):
        #check arguments
        pass

        
class BaseVD(BaseCoder):
    """
    BaseVD provides functionality for encoding and decoding data using Value Differencing steganography.
    It is an adapted and generalised version of the Pixel Value Differencing method as proposed by
    Wu, D. C., & Tsai, W. H. (2003).
    A steganographic method for images by pixel-value differencing.
    Pattern recognition letters, 24(9-10), 1613-1626.
    
    :param obj: The data object (NumPy array) where the message will be encoded/decoded.
    :type obj: numpy.ndarray
    :param pos_dim: number of dimensions without depth dimension
    :type pos_dim: int
    :param depth: number of channels per value
    :type depth: int
    """
    def __init__(self, obj, pos_dim=2, depth=3):
        super().__init__()
        self.data = obj
        self.pos_dim = pos_dim
        self.depth = depth
        self._range_offset = 3
        self._pixel_range_start = 1
        self.ranges = self.define_range(self.data.dtype)

    def define_range(self, dtype):
        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
        elif np.issubdtype(dtype, np.floating):
            info = np.finfo(dtype)
        else:
            raise Exception('array dtype must be integer or float.')
        
        min_value, max_value = info.min, info.max
        max_value += abs(min_value)
        min_value = 0
        lower = 2**self._range_offset - 1
        ranges = [(self._pixel_range_start,0,lower)]
        lower += 1
        n = 1 + self._range_offset
        while lower <= max_value:
            upper = min((2**n)-1,max_value)
            ranges.append((n-self._range_offset+self._pixel_range_start,lower,upper))
            lower = upper + 1
            n += 1       
        return ranges
    
    def _range(self, diff):
        """Return the number of bits to embed and the lower and upper bounds based on the pixel difference."""
        #faster then dictionary access
        for i, l_i, u_i in self.ranges:
            if l_i<=diff<=u_i:
                return i, l_i, u_i
            
    def _get_pairs(self, seed=None):
        indices = np.ndindex(*self.data.shape[:self.pos_dim])
        indices = list(indices)
        if seed is not None:
            np.random.default_rng(seed).shuffle(indices)
        pairs = [(indices[i], indices[i+1]) for i in range(0, len(indices)-1,2)]
        return pairs
    def max_capacity(self):
        return None
    
    def encode(self, message, seed=None,
               method='metadata', metadata_length=METADATA_LENGTH_IMAGE,
               delimiter_message=DELIMITER_MESSAGE, compress=False):
        """
        Encodes a message from the cover data using Value Differencing steganography.

        The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
        supports optional message verification to ensure the decoded message matches the original encoded message.
        :param message: The message to be hidden. Gets converted into binary if not already.
        :type message: str
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
        message = self._encode_message(message, method, metadata_length, delimiter_message, compress)
        value_pairs = self._get_pairs(seed)
        message_index = 0
        max_len = len(message)
        
        for (v1_coords, v2_coords) in value_pairs:
            if message_index >= max_len:
                break
            for channel in range(self.depth):
                v1 = int(self.data[*v1_coords,channel])
                v2 = int(self.data[*v2_coords,channel])
                d = v2-v1
                abs_d = abs(d)

                try:
                    bits_to_hide, l_i, u_i = self._range(abs_d)
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
                        bits_remaining = max_len - message_index
                        bits_to_encode = min(bits_to_hide, bits_remaining)
                        bin_secret = message[message_index:message_index + bits_to_encode]
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
                        message_index += bits_to_hide
                        
                        self.data[*v1_coords, channel] = v1_prime
                        self.data[*v2_coords, channel] = v2_prime
                    else:
                        
                        continue  # Skip if pixel values go out of bounds
        return True
    def decode(self, seed=None, method='metadata', metadata_length=METADATA_LENGTH_LSB,
               delimiter_message=DELIMITER_MESSAGE, compress=False):
        """
        Decodes a message from the cover data using Value Differencing steganography.

        The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
        supports optional message verification to ensure the decoded message matches the original encoded message.

        :param seed: (Optional) Seed value for pseudo-randomly distributing the message in the cover data.
        :type seed: int, optional
        :param method: Method for marking the end of the message. Options are 'delimiter', 'metadata', or None. Defaults to 'metadata'.
        :type method: str, optional
        :param metadata_length: Length of the metadata in bits when `method='metadata'`. Defaults to `METADATA_LENGTH_LSB`.
        :type metadata_length: int, optional
        :param delimiter_message: The delimiter string used when `method='delimiter'`.
        :type delimiter_message: str, optional

        :return: The decoded message
        :rtype: str or tuple
        """
        bin_message = ""
        value_pairs = self._get_pairs(seed)
        for (v1_coords, v2_coords) in value_pairs:
            for channel in range(self.depth):
                v1 = int(self.data[*v1_coords,channel])
                v2 = int(self.data[*v2_coords,channel])
                d = v2 - v1
                abs_d = abs(d)

                try:
                    bits_retrieved, l_i, u_i = self._range(abs_d)
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
                        bin_message += bits
                    else: 
                        continue  # Skip if pixel values go out of bounds
        
        if method == 'metadata':
            length = int(bin_message[:metadata_length], 2)
            message_end = metadata_length + length
            message = bin_message[metadata_length:message_end]
        elif method == 'delimiter':
            if not is_binary(delimiter_message):
                delimiter_message = data_to_binary(delimiter_message)
            message = bin_message.split(delimiter_message)[0]
        elif method is None:
            return bin_message
        else:
            raise Exception('Method must be either delimiter, metadata, or None.')

        if compress:
            message = compression.decompress(message)
        
        return message

    
class BaseLSB(BaseCoder):
    """
    BaseLSB provides functionality for encoding and decoding data using Least Significant Bit (LSB) steganography.

    :param obj: The data object (NumPy array) where the message will be encoded/decoded.
    :type obj: numpy.ndarray
    """
    def __init__(self, obj):
        """
        Initializes the BaseLSB encoder/decoder.

        :param obj: The data object (NumPy array) where the message will be encoded or decoded.
        :type obj: numpy.ndarray
        """
        super().__init__()
        self.data = obj.flatten().astype(np.int32)
    def max_capacity(self, bits=1):
        """
        Calculates the maximum capacity of the cover object for embedding a message.

        The capacity is determined by the size of the data array and the number of bits available for modification.

        :param bits: Number of bits changed per value. Defaults to 1.
        :type bits: int
        :return: The maximum capacity of the object in bits.
        :rtype: int
        """
        return len(self.data) * bits
    def encode(self, message, matching=False, seed=None, bits=1,
               method='metadata', metadata_length=METADATA_LENGTH_LSB,
               delimiter_message=DELIMITER_MESSAGE, compress=False):
        """
        Encodes a message into the cover data using LSB steganography.

        The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
        supports optional message verification to ensure the decoded message matches the original encoded message.
        :param message: The message to be hidden. Gets converted into binary if not already.
        :type message: str
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
        
        message = self._encode_message(message, method, metadata_length, delimiter_message, compress)
        if len(message) > self.max_capacity(bits):
            warnings.warn("Insufficient bits, need larger cover or smaller message.")

        mask = (1<<bits)-1
        message_array = utils.parse_message(message, bits)
        message_length = len(message_array)
        if seed:
            indices = prng_indices(len(self.data),seed)[:message_length]
        else:
            indices = np.arange(message_length)
        self.data[indices[:message_length]] &= ~mask
        self.data[indices[:message_length]] |= message_array
        return True
    def decode(self, matching=False, seed=None, bits=1,
               method='metadata', metadata_length=METADATA_LENGTH_LSB,
               delimiter_message=DELIMITER_MESSAGE, compress=False):
        """
        Decodes a message from the cover data using LSB steganography.

        The message can be decoded using either a delimiter, metadata, or without any end-of-message marker. It also
        supports optional message verification to ensure the decoded message matches the original encoded message.

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
        :param compress: Whether compression was used on the encoded data. Defaults to False.
        :type compress: bool, optional

        :return: The decoded message
        :rtype: str or tuple
        """
        if matching is not False: raise NotImplementedError("LSB matching not implemented in current version.")
        
        mask = (1<<bits)-1
        message_bits = ''
        if seed is not None:
            indices = prng_indices(len(self.data), seed)
        else:
            indices = np.arange(len(self.data))
        if method is None:
            extracted_bits = self.data[indices[:len(self.data)]] & mask
            message_bits = ''.join(f'{bit:0{bits}b}' for bit in extracted_bits)
        
        elif method == 'metadata':
            total_metadata_pixels = math.ceil(metadata_length / bits)
            metadata_bits = ''.join(f'{(self.data[indices[i]] & mask):0{bits}b}' for i in range(total_metadata_pixels))
            metadata_bits = metadata_bits[:metadata_length]
            message_length = int(metadata_bits, 2)
            total_message_pixels = math.ceil(message_length / bits)
            extracted_bits = self.data[indices[total_metadata_pixels:total_metadata_pixels + total_message_pixels]] & mask
            message_bits = utils.generate_message_bits(extracted_bits, bits)
            message_bits = message_bits[:message_length]
    
        elif method == 'delimiter':
            if not is_binary(delimiter_message):
                delimiter = data_to_binary(delimiter_message)
            
            for frame in self.data[indices]:
                extracted_bit = frame & mask
                message_bits += f'{extracted_bit:0{bits}b}'
                if message_bits.endswith(delimiter):
                    message_bits = message_bits[:-len(delimiter)]
                    break
        if compress:
            message_bits = compression.decompress(message_bits)
        return message_bits
