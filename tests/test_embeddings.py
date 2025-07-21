import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os

import stegosphere
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere.embeddings import LSB, VD, BPCS, echo_hiding
from stegosphere import io, payloader, selector


# =============================================================================
# LSB TESTS
# =============================================================================

class TestLSB:
    """Test cases for LSB steganography"""
    
    def test_max_capacity(self, generate_image):
        """Test capacity calculation"""
        capacity = LSB.max_capacity(generate_image, bits=1)
        expected = generate_image.size * 1
        assert capacity == expected
        
        capacity_2bit = LSB.max_capacity(generate_image, bits=2)
        expected_2bit = generate_image.size * 2
        assert capacity_2bit == expected_2bit

    def test_embed_extract_basic(self, generate_image, small_payload):
        """Test basic embed and extract functionality"""
        stego = LSB.embed(generate_image, small_payload)
        extracted = LSB.extract(stego)
        assert extracted == small_payload

    def test_embed_extract_with_seed(self, generate_image, small_payload):
        """Test embed/extract with PRNG seed"""
        seed = 1
        stego = LSB.embed(generate_image, small_payload, seed=seed)
        extracted = LSB.extract(stego, seed=seed)
        assert extracted == small_payload

    def test_embed_extract_custom_indices(self, generate_image, small_payload):
        """Test embed/extract with custom indices"""
        indices = np.arange(len(small_payload))
        stego = LSB.embed(generate_image, small_payload, framing=None, indices=indices)
        extracted = LSB.extract(stego, framing=None, indices=indices)
        assert extracted == small_payload

    def test_embed_multiple_bits(self, generate_image, small_payload):
        """Test embedding with multiple bits per pixel"""
        for bits in [1, 2, 3, 4]:
            stego = LSB.embed(generate_image, small_payload, bits=bits)
            extracted = LSB.extract(stego, bits=bits)
            assert extracted == small_payload

    @patch('stegosphere.embeddings.LSB.BACKEND', False)
    def test_python_backend_matching(self, generate_image):
        """Test LSB matching with Python backend"""
        payload = "10101010"
        stego = LSB.embed(generate_image, payload, matching=True)
        extracted = LSB.extract(stego)
        assert extracted == payload

    def test_insufficient_capacity_warning(self, generate_grayscale_image):
        """Test warning when payload is too large"""
        large_payload = "1" * (generate_grayscale_image.size * 2)
        with pytest.warns(UserWarning, match="Insufficient capacity"):
            LSB.embed(generate_grayscale_image, large_payload)

    def test_invalid_indices_error(self, generate_image):
        """Test error when indices are insufficient"""
        payload = "1010101010"
        short_indices = np.array([0, 1, 2])
        with pytest.raises(ValueError, match="indices too short"):
            LSB.embed(generate_image, payload, indices=short_indices)

    def test_matching_with_multiple_bits_error(self, generate_image):
        """Test error when using matching with multiple bits"""
        with patch('stegosphere.embeddings.LSB.BACKEND', False):
            with pytest.raises(NotImplementedError):
                LSB.embed(generate_image, "1010", matching=True, bits=2)

    def test_different_framing_methods(self, generate_image, small_payload):
        """Test different payload framing methods"""
        framings = [(payloader.FixedHeader, {}),
            (payloader.Delimiter, {}),
            (payloader.Unary, {}),
            (payloader.EliasGamma, {})]
        
        for framing, kwargs in framings:
            stego = LSB.embed(generate_image, small_payload, framing=framing, framing_kwargs=kwargs)
            extracted = LSB.extract(stego, framing=framing, framing_kwargs=kwargs)
            assert extracted == small_payload


# =============================================================================
# VALUE DIFFERENCING TESTS
# =============================================================================

class TestVD:
    """Test cases for Value Differencing steganography"""
    
    def test_embed_extract_basic(self, generate_image, small_payload):
        """Test basic VD embed and extract"""
        stego = VD.embed(generate_image, small_payload)
        extracted = VD.extract(stego)
        assert extracted == small_payload

    def test_define_range(self):
        """Test range definition"""
        ranges = VD.define_range()
        assert isinstance(ranges, list)
        assert len(ranges) > 0
        # Check format: (bits, lower, upper)
        for r in ranges:
            assert len(r) == 3
            assert all(isinstance(x, int) for x in r)

    def test_define_range_table(self):
        """Test range table creation"""
        ranges = VD.define_range()
        table = VD.define_range_table(ranges)
        assert isinstance(table, list)
        assert len(table) > 0

    def test_custom_ranges(self, generate_image, small_payload):
        """Test VD with custom ranges"""
        ranges = [(1, 0, 7), (2, 8, 15), (3, 16, 31)]
        stego = VD.embed(generate_image, small_payload, ranges=ranges)
        extracted = VD.extract(stego, ranges=ranges)
        assert extracted == small_payload

    def test_with_seed(self, generate_image, small_payload):
        """Test VD with PRNG seed"""
        seed = 123
        stego = VD.embed(generate_image, small_payload, seed=seed)
        extracted = VD.extract(stego, seed=seed)
        assert extracted == small_payload

    def test_different_dimensions(self, small_payload):
        """Test VD with different array dimensions"""
        # 1D array (audio-like)
        array_1d = np.random.randint(0, 256, 1000, dtype=np.uint8)
        stego = VD.embed(array_1d, small_payload)
        extracted = VD.extract(stego)
        assert extracted == small_payload
        
        # 2D grayscale
        array_2d = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        stego = VD.embed(array_2d, small_payload)
        extracted = VD.extract(stego)
        assert extracted == small_payload

    def test_framing_methods(self, generate_image, small_payload):
        """Test VD with different framing methods"""
        for framing in [payloader.FixedHeader, payloader.Delimiter]:
            stego = VD.embed(generate_image, small_payload, framing=framing)
            extracted = VD.extract(stego, framing=framing)
            assert extracted == small_payload


# =============================================================================
# BPCS TESTS  
# =============================================================================

class TestBPCS:
    """Test cases for BPCS steganography"""
    
    def test_embed_extract_basic(self, generate_image):
        """Test basic BPCS embed and extract"""
        payload = "1010101010101010"
        stego, conj_map, used_bits = BPCS.embed(generate_image, payload)
        extracted = BPCS.extract(stego, conj_map, len(payload))
        assert extracted == payload
        assert used_bits <= len(payload)

    def test_different_block_sizes(self, generate_grayscale_image):
        """Test BPCS with different block sizes"""
        payload = "10101010"
        for block_size in [4, 8, 16]:
            stego, conj_map, used_bits = BPCS.embed(generate_grayscale_image, payload, block_size=block_size)
            extracted = BPCS.extract(stego, conj_map, len(payload), block_size=block_size)
            assert extracted == payload

    def test_different_thresholds(self, generate_grayscale_image):
        """Test BPCS with different complexity thresholds"""
        payload = "11110000"
        for threshold in [0.1, 0.3, 0.5]:
            stego, conj_map, used_bits = BPCS.embed(generate_grayscale_image, payload, threshold=threshold)
            extracted = BPCS.extract(stego, conj_map, len(payload))
            assert extracted == payload

    def test_grayscale_and_color(self, generate_image, generate_grayscale_image):
        """Test BPCS with both grayscale and color images"""
        payload = "1111000010101010"
        
        # Color image
        stego_color, conj_map_color, used_bits = BPCS.embed(generate_image, payload)
        extracted_color = BPCS.extract(stego_color, conj_map_color, len(payload))
        assert extracted_color == payload
        
        # Grayscale image
        stego_gray, conj_map_gray, used_bits = BPCS.embed(generate_grayscale_image, payload)
        extracted_gray = BPCS.extract(stego_gray, conj_map_gray, len(payload))
        assert extracted_gray == payload

    def test_complexity_computation(self):
        """Test complexity computation function"""
        # High complexity block (checkerboard)
        high_complex = np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        complexity_high = BPCS._compute_complexity(high_complex)
        
        # Low complexity block (uniform)
        low_complex = np.ones((4, 4))
        complexity_low = BPCS._compute_complexity(low_complex)
        
        assert complexity_high > complexity_low
        assert 0 <= complexity_high <= 1
        assert 0 <= complexity_low <= 1

    def test_bitplane_operations(self, generate_grayscale_image):
        """Test bitplane split and reconstruction"""
        original = generate_grayscale_image
        bitplanes = BPCS._split_into_bitplanes(original)
        reconstructed = BPCS._reconstruct_from_bitplanes(bitplanes)
        
        assert len(bitplanes) == 8
        np.testing.assert_array_equal(original, reconstructed)


# =============================================================================
# ECHO HIDING TESTS
# =============================================================================

class TestEchoHiding:
    """Test cases for echo hiding steganography"""
    
    def test_echo_kernel(self):
        """Test echo kernel generation"""
        kernel = echo_hiding.echo_kernel(delay=100, alpha=0.5, length=200)
        assert len(kernel) == 200
        assert kernel[0] == 1.0
        assert kernel[100] == 0.5
        assert kernel[50] == 0.0

    def test_embed_extract_basic(self, generate_audio):
        """Test basic echo hiding embed and extract"""
        bits = "101010"
        stego = echo_hiding.embed(generate_audio, bits)
        extracted = echo_hiding.extract(stego, len(bits))
        assert extracted == bits

    def test_embed_extract_stereo(self, generate_stereo_audio):
        """Test echo hiding with stereo audio"""
        bits = "1100"
        stego = echo_hiding.embed(generate_stereo_audio, bits)
        extracted = echo_hiding.extract(stego, len(bits))
        assert extracted == bits

    def test_different_parameters(self, generate_audio):
        """Test echo hiding with different parameters"""
        bits = "1010"
        params = [
            {'delay0': 50, 'delay1': 150, 'alpha': 0.2},
            {'delay0': 200, 'delay1': 400, 'alpha': 0.4},
        ]
        
        for param in params:
            stego = echo_hiding.embed(generate_audio, bits, **param)
            extracted = echo_hiding.extract(stego, len(bits), **param)
            assert extracted == bits

    def test_delay_detection(self, generate_audio):
        """Test delay detection function"""
        frame = np.random.randn(1024)
        delays = (100, 200)
        detected = echo_hiding.detect_delay(frame, delays, alpha=0.3)
        assert detected in delays

    def test_frame_size_variations(self, generate_audio):
        """Test different frame sizes"""
        bits = "10"
        for frame_size in [512, 1024, 2048]:
            if len(generate_audio) >= frame_size * len(bits):
                stego = echo_hiding.embed(generate_audio, bits, frame_size=frame_size)
                extracted = echo_hiding.extract(stego, len(bits), frame_size=frame_size)
                assert extracted == bits