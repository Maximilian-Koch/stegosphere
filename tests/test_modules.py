#testing the selector, payloader, and io module
import pytest
import numpy as np
import tempfile
import os
import warnings

from stegosphere import io, payloader, selector
from stegosphere.utils import generate_binary_payload as gbp


# =============================================================================
# IO MODULE TESTS
# =============================================================================

class TestIO:
    """Test cases for IO module"""
    
    def test_data_to_binary_conversions(self, text_payload):
        """Test data to binary conversions"""
        # String to binary
        binary = io.data_to_binary(text_payload)
        assert isinstance(binary, str)
        assert io.is_binary(binary)
        
        # Bytes to binary
        bytes_data = text_payload.encode('utf-8')
        binary_from_bytes = io.data_to_binary(bytes_data)
        assert binary == binary_from_bytes
        
        # Integer to binary
        int_binary = io.data_to_binary(65)  # 'A'
        assert int_binary == '01000001'

    def test_binary_to_data(self, text_payload):
        """Test binary to data conversion"""
        binary = io.data_to_binary(text_payload)
        recovered = io.binary_to_data(binary)
        assert recovered.decode('utf-8') == text_payload

    def test_hex_to_binary(self, hex_payload):
        """Test hex to binary conversion"""
        binary = io.hex_to_binary(hex_payload)
        assert io.is_binary(binary)

    def test_check_type(self, binary_payload, hex_payload, text_payload):
        """Test type checking function"""
        assert io.check_type(binary_payload) == 0  # binary
        assert io.check_type(hex_payload) == 1     # hex
        assert io.check_type(text_payload) == 2    # string

    def test_is_binary(self, binary_payload):
        """Test binary string detection"""
        assert io.is_binary(binary_payload) == True
        assert io.is_binary("hello") == False
        assert io.is_binary("102010") == False

    def test_encode_decode_binary_payload(self, text_payload):
        """Test payload encoding and decoding"""
        # Test with different framings
        framings = [
            (payloader.FixedHeader, {}),
            (payloader.RawData, {}),
            (None, {})
        ]
        
        for framing, kwargs in framings:
            encoded = io.encode_binary_payload(text_payload, framing, kwargs)
            decoded = io.decode_binary_payload(encoded, framing, kwargs)
            assert decoded == io.data_to_binary(text_payload)

    def test_file_operations(self, text_payload):
        """Test file read/write operations"""
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(text_payload.encode('utf-8'))
            tmp_file_path = tmp_file.name
        
        try:
            # Read file to binary
            binary_data = io.file_to_binary(tmp_file_path)
            assert io.is_binary(binary_data)
            
            # Write binary to new file
            output_path = tmp_file_path + "_output"
            io.binary_to_file(binary_data, output_path)
            
            # Verify content
            with open(output_path, 'r', encoding='utf-8') as f:
                recovered_text = f.read()
            assert recovered_text == text_payload
            
        finally:
            # Cleanup
            os.unlink(tmp_file_path)
            if os.path.exists(output_path):
                os.unlink(output_path)


# =============================================================================
# PAYLOADER TESTS
# =============================================================================

class TestPayloader:
    """Test cases for payloader module"""
    
    def test_raw_data_framing(self, binary_payload):
        """Test raw data framing (no framing)"""
        encoded = payloader.RawData.encode(binary_payload)
        decoded = payloader.RawData.decode(encoded)
        assert encoded == binary_payload
        assert decoded == binary_payload

    def test_fixed_length_framing(self, binary_payload):
        """Test fixed length framing"""
        n_bits = len(binary_payload) // 2
        encoded = payloader.FixedLength.encode(binary_payload, n_bits=n_bits)
        decoded = payloader.FixedLength.decode(encoded, n_bits=n_bits)
        assert encoded == binary_payload
        assert decoded == binary_payload[:n_bits]

    def test_fixed_header_framing(self, binary_payload):
        """Test fixed header framing"""
        encoded = payloader.FixedHeader.encode(binary_payload)
        decoded = payloader.FixedHeader.decode(encoded)
        assert decoded == binary_payload
        
        # Test custom header length
        encoded_custom = payloader.FixedHeader.encode(binary_payload, fixed_header_length=16)
        decoded_custom = payloader.FixedHeader.decode(encoded_custom, fixed_header_length=16)
        assert decoded_custom == binary_payload

    def test_delimiter_framing(self, binary_payload):
        """Test delimiter framing"""
        encoded = payloader.Delimiter.encode(binary_payload)
        decoded = payloader.Delimiter.decode(encoded)
        assert decoded == binary_payload
        
        # Test custom sentinel
        custom_sentinel = "0110100101010"
        encoded_custom = payloader.Delimiter.encode(binary_payload, sentinel=custom_sentinel)
        decoded_custom = payloader.Delimiter.decode(encoded_custom, sentinel=custom_sentinel)
        assert decoded_custom == binary_payload

    def test_unary_framing(self, binary_payload):
        """Test unary framing"""
        encoded = payloader.Unary.encode(binary_payload)
        decoded = payloader.Unary.decode(encoded)
        assert decoded == binary_payload

    def test_elias_gamma_framing(self, binary_payload):
        """Test Elias Gamma framing"""
        encoded = payloader.EliasGamma.encode(binary_payload)
        decoded = payloader.EliasGamma.decode(encoded)
        assert decoded == binary_payload

    def test_framing_errors(self):
        """Test error conditions in framing"""
        # FixedLength without n_bits
        with pytest.raises(ValueError):
            payloader.FixedLength.decode("1010", n_bits=None)
        
        # Delimiter without sentinel
        with pytest.raises(ValueError):
            payloader.Delimiter.decode("1010", sentinel="1111")
        
        # Unary without terminator
        with pytest.raises(ValueError):
            payloader.Unary.decode("1111")
        
        # EliasGamma with invalid payload
        with pytest.raises(ValueError):
            payloader.EliasGamma.encode("")

    def test_encode_decode_framing_functions(self, binary_payload):
        """Test top-level encode/decode framing functions"""
        framings = [
            payloader.FixedHeader,
            payloader.Delimiter,
            payloader.Unary,
            payloader.EliasGamma
        ]
        
        for framing in framings:
            encoded = payloader.encode_framing(binary_payload, framing)
            decoded = payloader.decode_framing(encoded, framing)
            assert decoded == binary_payload


# =============================================================================
# SELECTOR TESTS
# =============================================================================

class TestSelector:
    """Test cases for selector module"""
    
    def test_greedy_indices(self):
        """Test greedy index selection"""
        costs = np.array([0.8, 0.2, 0.5, 0.1, 0.9])
        indices = selector.greedy_indices(costs, n=3)
        expected_order = [3, 1, 2]  # Sorted by cost: 0.1, 0.2, 0.5
        np.testing.assert_array_equal(indices, expected_order)
        
        # Test all indices
        all_indices = selector.greedy_indices(costs, n=-1)
        assert len(all_indices) == len(costs)

    def test_prng_indices(self):
        """Test PRNG index generation"""
        size = 100
        seed = 1
        
        # Test with integer seed
        indices1 = selector.prng_indices(size, seed)
        indices2 = selector.prng_indices(size, seed)
        np.testing.assert_array_equal(indices1, indices2) 

    def test_prng_indices_choice(self):
        """Test PRNG index selection with choice"""
        size = 1000
        length = 100
        seed = 2
        
        indices = selector.prng_indices_choice(size, seed, length)
        assert len(indices) == length
        assert len(np.unique(indices)) == length  # All unique
        assert np.all(indices < size)  # Within bounds

    def test_weighted_greedy_prng_indices(self):
        """Test weighted PRNG index selection"""
        max_index = 100
        weights = np.random.rand(max_index)
        n = 20
        seed = 3
        
        indices = selector.weighted_greedy_prng_indices(max_index, weights, n, seed)
        assert len(indices) == n
        assert len(np.unique(indices)) == n  # All unique
        assert np.all(indices < max_index)  # Within bounds

    def test_inverse_cost_greedy_prng_indices(self):
        """Test inverse cost weighted selection"""
        costs = np.array([1.0, 2.0, 0.5, 3.0, 0.1])  # Lower costs should be more likely
        n = 3
        seed = 4
        
        indices = selector.inverse_cost_greedy_prng_indices(costs, n, seed)
        assert len(indices) == n
        assert len(np.unique(indices)) == n

