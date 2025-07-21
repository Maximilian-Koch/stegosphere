import numpy as np
import warnings
import pytest

import stegosphere
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere.transform import IWT

# =============================================================================
# IWT (INTEGER WAVELET TRANSFORM) TESTS
# =============================================================================

class TestIWT:
    """Test cases for Integer Wavelet Transform"""
    
    def test_transform_inverse_basic(self, generate_grayscale_image):
        """Test basic transform and inverse"""
        coeffs, boundary_info = IWT.transform(generate_grayscale_image)
        reconstructed = IWT.inverse(coeffs, boundary_info)
        
        # Should be able to reconstruct exactly (integer transform)
        np.testing.assert_array_equal(generate_grayscale_image, reconstructed)

    def test_transform_color_image(self, generate_image):
        """Test transform with color image (skip last axis)"""
        coeffs, boundary_info = IWT.transform(generate_image, skip_last_axis=True)
        reconstructed = IWT.inverse(coeffs, boundary_info)
        
        np.testing.assert_array_equal(generate_image, reconstructed)

    def test_different_boundary_modes(self, generate_grayscale_image):
        """Test different boundary handling modes"""
        # Make image with odd dimensions
        odd_image = generate_grayscale_image[:99, :99]  # 99x99
        
        modes = [IWT.MODE_DISCARD, IWT.MODE_SYMMETRIC, IWT.MODE_REFLECT, 
                IWT.MODE_EDGE, IWT.MODE_WRAP]
        
        for mode in modes:
            if mode != IWT.MODE_DISCARD:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")  # Ignore boundary mode warnings
                    coeffs, boundary_info = IWT.transform(odd_image, boundary_mode=mode)
                    reconstructed = IWT.inverse(coeffs, boundary_info)
                    np.testing.assert_array_equal(odd_image, reconstructed)
            else:
                coeffs, boundary_info = IWT.transform(odd_image, boundary_mode=mode)
                reconstructed = IWT.inverse(coeffs, boundary_info)
                np.testing.assert_array_equal(odd_image, reconstructed)

    def test_1d_array(self):
        """Test IWT with 1D array"""
        array_1d = np.random.randint(0, 256, 64, dtype=np.uint8)
        coeffs, boundary_info = IWT.transform(array_1d)
        reconstructed = IWT.inverse(coeffs, boundary_info)
        
        np.testing.assert_array_equal(array_1d, reconstructed)

    def test_coefficients_structure(self, generate_grayscale_image):
        """Test the structure of wavelet coefficients"""
        coeffs, boundary_info = IWT.transform(generate_grayscale_image)
        
        # Coefficients should be a dictionary
        assert isinstance(coeffs, dict)
        assert len(coeffs) > 0
        
        # Keys should be tuples of '0' and '1'
        for key in coeffs.keys():
            assert isinstance(key, tuple)
            for k in key:
                assert k in ['0', '1']

    def test_error_conditions(self):
        """Test error conditions"""
        # Invalid array type
        with pytest.raises(TypeError):
            IWT.transform("not an array")
        
        # Invalid boundary mode
        array = np.random.randint(0, 256, (32, 32), dtype=np.uint8)
        with pytest.raises(ValueError):
            IWT.transform(array, boundary_mode="invalid_mode")
        
        # Invalid coefficients for inverse
        with pytest.raises(TypeError):
            IWT.inverse("not a dict", {'mode': 'discard', 'details': {}})

    def test_even_dimensions(self):
        """Test with even dimensions (should work without boundary handling)"""
        even_array = np.random.randint(0, 256, (64, 64), dtype=np.uint8)
        coeffs, boundary_info = IWT.transform(even_array)
        reconstructed = IWT.inverse(coeffs, boundary_info)
        
        np.testing.assert_array_equal(even_array, reconstructed)