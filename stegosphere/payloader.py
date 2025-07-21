"""
Payload framing utilities for binary data encoding.

Defines framing schemes for prefixing or delimiting binary payloads 
to allow reliable extraction. Includes abstract base class `PayloadFrame` 
and implementations:
- RawData (no framing)
- FixedLength (known payload size)
- FixedHeader (length prefix)
- Delimiter (sentinel-based)
- Unary and EliasGamma (self-delimiting codes)

Used for lossless transmission or embedding of binary payloads.
"""

import math
from abc import ABC, abstractmethod
import warnings

__all__ = ['FIXED_HEADER_LENGTH_DEFAULT', 'SENTINEL_DEFAULT', 'encode_framing', 'decode_framing',
           'PayloadFrame', 'RawData', 'FixedLength', 'FixedHeader',
           'Delimiter', 'Unary', 'EliasGamma']

FIXED_HEADER_LENGTH_DEFAULT = 32
SENTINEL_DEFAULT = '0010001100100010001000110010001100100010001000110010001100100011'

def encode_framing(payload, framing=None, framing_kwargs=None):
    """
    Encodes a binary payload using specified framing method.

    Args:
        payload (str): Payload to be encoded
        framing (stegosphere.payloader.PayloadFrame | None, optional): Framing method to use
        framing_kwargs (dict | None): specific framing arguments
    Returns:
        str: Encoded binary payload
    """
    framing_kwargs = framing_kwargs or {}
    if framing is None:
        return payload
    if not isinstance(payload, str):
        raise TypeError(f"Payload must be a binary string, got {type(payload).__name__}")

    return framing.encode(payload, **framing_kwargs)

def decode_framing(payload, framing=None, framing_kwargs=None):
    """
    Decodes payload by removing payload framing.

    Args:
        payload (str): Payload to be decoded
        framing (stegosphere.payloader.PayloadFrame | None, optional): Framing method to use
        framing_kwargs (dict | None): specific framing arguments
    Returns:
        str: Decoded binary payload
    """
    framing_kwargs = framing_kwargs or {}
    if framing is None:
        return payload
    if not isinstance(payload, str):
        raise TypeError(f"Framed payload must be a binary string, got {type(payload).__name__}")
    
    return framing.decode(payload, **framing_kwargs)

class PayloadFrame(ABC):
    """
    Abstract base class for payload framing methods.
    Subclasses must implement static encode() and decode() methods.
    """
    @staticmethod
    @abstractmethod
    def encode(payload: str, **kwargs) -> str:
        """
        Frame and encode a binary payload string.

        Args:
            payload (str): Payload string, often binary
            **kwargs: Method-specific parameters.

        Returns:
            str: The framed string.
        """
        pass

    @staticmethod
    @abstractmethod
    def decode(payload: str, **kwargs) -> str:
        """
        Decode a framed payload and extract the original data bits.

        Args:
            payload (str): A framed string to decode.
            **kwargs: Method-specific parameters.

        Returns:
            str: The original payload.
        """
        pass

    def __str__(self):
        return self.__class__.__name__

    def __repr__(self):
        return f"<{self.__class__.__name__}>"


class RawData(PayloadFrame):
    """
    Equivalent to not using any payload framing
    """
    @staticmethod
    def encode(payload):
        return payload
    @staticmethod
    def decode(payload):
        return payload

class FixedLength(PayloadFrame):
    """
    Fixed length is needed in decoding.
    """
    @staticmethod
    def encode(payload, n_bits=None, **kwargs):
        return payload
    @staticmethod
    def decode(payload: str, n_bits=None, **kwargs) -> str:
        if n_bits is None:
            raise ValueError("FixedLength.decode requires n_bits")
        if not isinstance(n_bits, int) or n_bits < 0:
            raise TypeError("n_bits must be a non‑negative integer")
        return payload[:n_bits]


class FixedHeader(PayloadFrame):
    """
    Fixed-header framing prefixes the payload with a binary length of fixed bit width.
    Use `fixed_header_length` in **kwargs to specify length.
    Defaults to `stegosphere.payloader.FIXED_HEADER_LENGTH_DEFAULT`.
    32 bit for metadata length should be sufficient for almost all usecases (payload size below 0.5GB).
    """
    @staticmethod
    def encode(payload, **kwargs):
        """
        Encode by prepending the payload length as a binary string of fixed width.

        Args:
            payload (str): Payload to be framed
            **kwargs: use `fixed_header_length` as additional argument
        
        Returns:
            str: Framed payload
        
        Raises:
            TypeError: If `fixed_header_length` is not a positive integer
        """
        fixed_header_length = kwargs.get('fixed_header_length', FIXED_HEADER_LENGTH_DEFAULT)
        if not isinstance(fixed_header_length, int) or fixed_header_length <= 0:
            raise TypeError("fixed_header_length must be a positive integer")
        
        return f"{len(payload):0{fixed_header_length}b}" + payload

    @staticmethod
    def decode(payload: str, **kwargs) -> str:
        """
        Decode by reading the fixed-width header, extracting that many bits as payload.

        Args:
            payload (str): Payload to be extracted from
            **kwargs: use `fixed_header_length` as additional argument
        
        Returns:
            str: Original payload
        
        Raises:
            ValueError: If payload is shorter than the `fixed_header_length`

        Warns:
            UserWarning: If framed payload is shorter than the indicated length. Will return entire payload.
        """
        fixed_header_length = kwargs.get('fixed_header_length', FIXED_HEADER_LENGTH_DEFAULT)
        if len(payload) < fixed_header_length:
            raise ValueError("Payload shorter than header length")
        length = int(payload[:fixed_header_length], 2)
        start = fixed_header_length
        end = start + length
        if len(payload) < end:
            warnings.warn("Framed payload is shorter than indicated length. Returning everything", UserWarning)
            return payload
        return payload[start:end]


class Delimiter(PayloadFrame):
    """
    Delimiter-based framing: appends a sentinel bit-pattern to mark the payload end.
    """
    @staticmethod
    def encode(payload, **kwargs) -> str:
        """
        Encode by appending the sentinel after the payload.

        Args:
            payload (str): Payload to be framed
            **kwargs: use `sentinel` as additional argument
        
        Returns:
            str: Framed payload
        """
        sentinel = kwargs.get('sentinel',SENTINEL_DEFAULT)
        return payload + sentinel

    @staticmethod
    def decode(payload, **kwargs):
        """
        Decode by splitting at the sentinel and returning data before it.

        Args:
            payload (str): Payload to be extracted from
            **kwargs: use `sentinel` as additional argument
        
        Returns:
            str: Original payload
        
        Raises:
            ValueError: If sentinel is not found in the payload
        
        Warns:
            UserWarning: If the sentinel occurs inside the payload data as well.
        """
        sentinel = kwargs.get('sentinel',SENTINEL_DEFAULT)
        if sentinel not in payload:
            raise ValueError("Sentinel not found in payload")
        parts = payload.split(sentinel, 1)
        result = parts[0]
        if sentinel in result:
            warnings.warn("Sentinel occurs inside payload data", UserWarning)
        return result

class Unary(PayloadFrame):
    """
    Unary framing: prefixes the payload with a unary-coded length (n '1's and a '0').
    This self-delimiting code requires no extra parameters.
    """
    @staticmethod
    def encode(payload: str, **kwargs) -> str:
        """
        Encode by prefixing with unary length.

        Args:
            payload (str): Payload to be framed
        
        Returns:
            str: Framed payload
        """
        length = len(payload)
        return '1' * length + '0' + payload

    @staticmethod
    def decode(payload: str, **kwargs) -> str:
        """
        Decode by counting leading '1's up to the first '0', then reading that many bits.

        Args:
            payload (str): Payload to be extracted from
        Returns:
            str: Original payload
        Raises:
            ValueError: If terminator is not found
        """
        if '0' not in payload:
            raise ValueError("Unary framing: no terminator '0' found")
        n = payload.index('0')
        start = n + 1
        return payload[start:start + n]


class EliasGamma(PayloadFrame):
    """
    Elias Gamma framing: unary-encodes floor(log2(L)), appends binary suffix, then payload.
    No extra parameters needed.
    """
    @staticmethod
    def encode(payload: str, **kwargs) -> str:
        """
        Encode by:
         1. Let L = len(payload)
         2. Compute N = floor(log2(L))
         3. Unary prefix = '1'*N + '0'
         4. Binary suffix = binary(L) without leading '1'
         5. Result = prefix + suffix + payload

        Args:
            payload (str): Payload to be framed
        
        Returns:
            str: Framed payload

        Raises:
            ValueError: If length of payload is <=0
        """
        L = len(payload)
        if L <= 0:
            raise ValueError("Payload length must be > 0")
        N = L.bit_length() - 1
        unary = '1' * N + '0'
        binary_suffix = format(L, 'b')[1:]
        return unary + binary_suffix + payload

    @staticmethod
    def decode(payload: str, **kwargs) -> str:
        """
        Decode by:
         1. Read unary prefix to get N (count of leading '1's)
         2. Skip zero
         3. Read N bits as suffix, reconstruct L = 2^N + int(suffix,2)
         4. Read next L bits as payload

        Args:
            payload (str): Framed payload
        Returns:
            str: Original payload
        Raises:
            ValueError:

        """
        idx = 0
        while idx < len(payload) and payload[idx] == '1':
            idx += 1
        if idx >= len(payload) or payload[idx] != '0':
            raise ValueError("Invalid Elias‑Gamma prefix")
        N = idx
        idx += 1
        suffix = payload[idx:idx + N]
        idx += N
        L = (1 << N) + (int(suffix, 2) if suffix else 0)
        end = idx + L
        if end > len(payload):
            raise ValueError("Elias‑Gamma framed payload too short")
        return payload[idx:end]