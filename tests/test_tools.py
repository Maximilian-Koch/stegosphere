#currently only testing multifile.py
import pytest
import numpy as np
import warnings

import stegosphere
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere import selector
from stegosphere.tools import multifile

# =============================================================================
# MULTIFILE TESTS
# =============================================================================

class TestMultifile:
    """Test cases for multifile module"""
    
    def test_weighted_chunks_basic(self):
        """Test basic weighted chunk distribution"""
        payload = "1010101010101010"  # 16 bits
        weights = [0.5, 0.3, 0.2]
        chunks = multifile.weighted_chunks(payload, 3, weights)
        
        assert len(chunks) == 3
        assert ''.join(chunks) == payload
        # Check approximate chunk sizes
        assert len(chunks[0]) == 8  # 50% of 16
        assert len(chunks[1]) == 5  # 30% of 16 (rounded)
        assert len(chunks[2]) == 3  # 20% of 16 (with adjustment)

    def test_weighted_chunks_rounding_adjustment(self):
        """Test rounding adjustment in weighted chunks"""
        payload = "101010101"  # 9 bits
        weights = [0.33, 0.33, 0.34]
        chunks = multifile.weighted_chunks(payload, 3, weights)
        
        assert len(chunks) == 3
        assert sum(len(chunk) for chunk in chunks) == len(payload)
        assert ''.join(chunks) == payload

    def test_weighted_chunks_invalid_weights(self):
        """Test error conditions for weighted chunks"""
        payload = "1010"
        
        # Weights don't sum to 1
        with pytest.raises(ValueError):
            multifile.weighted_chunks(payload, 2, [0.6, 0.6])
        
        # Wrong number of weights
        with pytest.raises(ValueError):
            multifile.weighted_chunks(payload, 3, [0.5, 0.5])

    def test_roundrobin_chunks_basic(self):
        """Test round-robin chunk distribution"""
        payload = "123456789"
        chunks = multifile.roundrobin_chunks(payload, 3)
        
        assert len(chunks) == 3
        assert chunks[0] == "147"  # indices 0, 3, 6
        assert chunks[1] == "258"  # indices 1, 4, 7
        assert chunks[2] == "369"  # indices 2, 5, 8

    def test_roundrobin_uneven_distribution(self):
        """Test round-robin with uneven payload length"""
        payload = "12345"
        chunks = multifile.roundrobin_chunks(payload, 3)
        
        assert len(chunks) == 3
        assert chunks[0] == "14"   # indices 0, 3
        assert chunks[1] == "25"   # indices 1, 4
        assert chunks[2] == "3"    # index 2

    def test_reverse_roundrobin_basic(self):
        """Test reverse round-robin reconstruction"""
        original = "123456789"
        chunks = multifile.roundrobin_chunks(original, 3)
        reconstructed_payload = ''.join(chunks)
        reversed_payload = multifile.reverse_roundrobin(reconstructed_payload, 3)
        
        assert reversed_payload == original

    def test_reverse_roundrobin_uneven(self):
        """Test reverse round-robin with uneven distribution"""
        original = "12345"
        chunks = multifile.roundrobin_chunks(original, 3)
        reconstructed_payload = ''.join(chunks)
        reversed_payload = multifile.reverse_roundrobin(reconstructed_payload, 3)
        
        assert reversed_payload == original

    def test_split_encode_even_distribution(self, small_payload):
        """Test split encode with even distribution"""
        # Mock instances that just return their input
        instances = [lambda x: x, lambda x: x, lambda x: x]
        
        outputs = multifile.split_encode(small_payload, instances, distribution='even')
        assert len(outputs) == 3
        assert ''.join(outputs) == small_payload

    def test_split_encode_weighted_distribution(self, small_payload):
        """Test split encode with weighted distribution"""
        instances = [lambda x: x, lambda x: x]
        weights = [0.7, 0.3]
        
        outputs = multifile.split_encode(
            small_payload, instances, 
            distribution='weighted', 
            distribution_args={'weights': weights}
        )
        
        assert len(outputs) == 2
        assert ''.join(outputs) == small_payload

    def test_split_encode_roundrobin_distribution(self, small_payload):
        """Test split encode with round-robin distribution"""
        instances = [lambda x: x, lambda x: x, lambda x: x]
        
        outputs = multifile.split_encode(small_payload, instances, distribution='roundrobin')
        assert len(outputs) == 3
        
        # Reconstruct using reverse round-robin
        combined = ''.join(outputs)
        reconstructed = multifile.reverse_roundrobin(combined, 3)
        assert reconstructed == small_payload

    def test_split_encode_with_seed(self, small_payload):
        """Test split encode with PRNG seed"""
        instances = [lambda x: x, lambda x: x]
        seed = 0
        
        outputs1 = multifile.split_encode(small_payload, instances, seed=seed)
        outputs2 = multifile.split_encode(small_payload, instances, seed=seed)
        
        assert outputs1 == outputs2  # Should be deterministic

    def test_split_encode_unknown_distribution(self, small_payload):
        """Test error for unknown distribution method"""
        instances = [lambda x: x]
        
        with pytest.raises(ValueError):
            multifile.split_encode(small_payload, instances, distribution='unknown')

    def test_split_decode_basic(self):
        """Test basic split decode functionality"""
        # Mock instances that return predefined chunks
        instances = [lambda: "101", lambda: "010", lambda: "111"]
        
        result = multifile.split_decode(instances)
        assert result == "101010111"

    def test_split_decode_roundrobin(self):
        """Test split decode with round-robin reconstruction"""
        # Create instances that return round-robin distributed chunks
        original = "123456789"
        chunks = multifile.roundrobin_chunks(original, 3)
        instances = [lambda chunk=chunk: chunk for chunk in chunks]
        
        result = multifile.split_decode(instances, distribution='roundrobin')
        assert result == original

    def test_split_decode_with_seed(self):
        """Test split decode with PRNG seed"""
        payload = "101010"
        seed = 0
        
        # Simulate encoding with seed (scrambles payload)
        indices = selector.prng_indices(len(payload), seed)
        scrambled = ''.join(payload[i] for i in indices)
        
        # Mock instances returning scrambled chunks
        instances = [lambda: scrambled]
        
        # Decode should unscramble
        result = multifile.split_decode(instances, seed=seed)
        assert result == payload