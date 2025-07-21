#testing the analysis modules
#only accuracy and robustness are currently being tested!
import pytest
import cv2
from scipy.signal import convolve2d
import numpy as np

from stegosphere.analysis import accuracy, robustness

# =============================================================================
# ACCURACY TESTS
# =============================================================================

class TestAccuracy:
    """Test cases for accuracy analysis module"""
    
    def test_hamming_distance_identical(self):
        """Test Hamming distance with identical sequences"""
        seq1 = "101010101"
        seq2 = "101010101"
        distance = accuracy.hamming_distance(seq1, seq2)
        assert distance == 0

    def test_hamming_distance_different(self):
        """Test Hamming distance with different sequences"""
        seq1 = "101010101"
        seq2 = "111010100"
        distance = accuracy.hamming_distance(seq1, seq2)
        assert distance == 2

    def test_hamming_distance_arrays(self):
        """Test Hamming distance with numpy arrays"""
        arr1 = np.array([1, 0, 1, 0, 1])
        arr2 = np.array([1, 1, 1, 0, 0])
        distance = accuracy.hamming_distance(arr1, arr2)
        assert distance == 2

    def test_hamming_distance_lists(self):
        """Test Hamming distance with lists"""
        list1 = [1, 0, 1, 0, 1]
        list2 = [0, 0, 1, 0, 1]
        distance = accuracy.hamming_distance(list1, list2)
        assert distance == 1

    def test_hamming_distance_length_mismatch(self):
        """Test error when sequences have different lengths"""
        seq1 = "1010"
        seq2 = "10101"
        with pytest.raises(ValueError, match="Inputs must be of equal length"):
            accuracy.hamming_distance(seq1, seq2)

    def test_bit_error_rate_zero(self):
        """Test bit error rate with identical sequences"""
        seq1 = "101010"
        seq2 = "101010"
        ber = accuracy.bit_error_rate(seq1, seq2)
        assert ber == 0.0

    def test_bit_error_rate_nonzero(self):
        """Test bit error rate with different sequences"""
        seq1 = "1010"
        seq2 = "1110"
        ber = accuracy.bit_error_rate(seq1, seq2)
        assert ber == 0.25

    def test_bit_error_rate_empty(self):
        """Test bit error rate with empty sequences"""
        ber = accuracy.bit_error_rate("", "")
        assert ber == 0.0

    def test_bit_error_rate_length_mismatch(self):
        """Test error when sequences have different lengths"""
        with pytest.raises(ValueError, match="Inputs must be of equal length"):
            accuracy.bit_error_rate("101", "1010")

    def test_normalized_correlation_identical(self):
        """Test normalized correlation with identical sequences"""
        seq = [1, 2, 3, 4, 5]
        corr = accuracy.normalized_correlation(seq, seq)
        assert corr == 1.0

    def test_normalized_correlation_orthogonal(self):
        """Test normalized correlation with orthogonal sequences"""
        seq1 = [1, -1, 1, -1]
        seq2 = [-1, 1, -1, 1]
        corr = accuracy.normalized_correlation(seq1, seq2)
        assert corr == -1.0

    def test_normalized_correlation_zero_denominator(self):
        """Test normalized correlation when denominator is zero"""
        seq1 = [0, 0, 0]
        seq2 = [1, 2, 3]
        corr = accuracy.normalized_correlation(seq1, seq2)
        assert corr == 0.0

    def test_normalized_correlation_different_types(self):
        """Test normalized correlation with different input types"""
        list1 = [1, 2, 3]
        array2 = np.array([2, 4, 6])
        corr = accuracy.normalized_correlation(list1, array2)
        expected = (1*2 + 2*4 + 3*6) / (1*1 + 2*2 + 3*3)
        assert np.isclose(corr, expected)

    def test_normalized_correlation_length_mismatch(self):
        """Test error when sequences have different lengths"""
        with pytest.raises(ValueError, match="Inputs must be of equal length"):
            accuracy.normalized_correlation([1, 2], [1, 2, 3])


# =============================================================================
# ROBUSTNESS TESTS
# =============================================================================

class TestRobustness:
    """Test cases for robustness analysis module"""
    
    def test_random_bit_attack_basic(self, generate_image):
        """Test basic random bit attack"""
        ratio = 0.1
        attacked = robustness.random_bit_attack(generate_image, ratio)
        
        # Should be different from original
        assert not np.array_equal(generate_image, attacked)
        # Should have same shape
        assert attacked.shape == generate_image.shape
        # Should be clipped to valid range
        assert np.all(attacked >= 0)
        assert np.all(attacked <= 255)

    def test_random_bit_attack_zero_ratio(self, generate_image):
        """Test random bit attack with zero ratio"""
        attacked = robustness.random_bit_attack(generate_image, ratio=0.0)
        np.testing.assert_array_equal(generate_image, attacked)

    def test_random_bit_attack_full_ratio(self, generate_image):
        """Test random bit attack with full ratio"""
        attacked = robustness.random_bit_attack(generate_image, ratio=1.0)
        # Should modify all pixels
        assert not np.array_equal(generate_image, attacked)

    def test_random_bit_attack_invalid_ratio(self, generate_image):
        """Test error for invalid ratio values"""
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            robustness.random_bit_attack(generate_image, ratio=-0.1)
        
        with pytest.raises(ValueError, match="ratio must be between 0 and 1"):
            robustness.random_bit_attack(generate_image, ratio=1.5)

    def test_random_bit_attack_float_array(self):
        """Test random bit attack on float array"""
        float_array = np.random.rand(10, 10).astype(np.float32)
        attacked = robustness.random_bit_attack(float_array, ratio=0.1)
        
        assert attacked.dtype == np.float32
        assert not np.array_equal(float_array, attacked)

    def test_gaussian_noise_attack_basic(self, generate_image):
        """Test Gaussian noise attack"""
        sigma = 1.0
        smoothed = robustness.gaussian_noise_attack(generate_image, sigma)
        
        # Should be different (smoothed)
        assert not np.array_equal(generate_image, smoothed)
        # Should have same shape and dtype
        assert smoothed.shape == generate_image.shape
        assert smoothed.dtype == generate_image.dtype

    def test_gaussian_noise_attack_different_sigmas(self, generate_grayscale_image):
        """Test Gaussian attack with different sigma values"""
        sigmas = [0.5, 1.0, 2.0]
        results = []
        
        for sigma in sigmas:
            smoothed = robustness.gaussian_noise_attack(generate_grayscale_image, sigma)
            results.append(smoothed)
        
        # Higher sigma should produce more smoothing
        # (compare variance as a measure of smoothness)
        variances = [np.var(result) for result in results]
        assert variances[0] > variances[1] > variances[2]

    def test_salt_and_pepper_attack_basic(self, generate_image):
        """Test salt and pepper attack"""
        attacked = robustness.salt_and_pepper_attack(generate_image, amount=0.01)
        
        assert not np.array_equal(generate_image, attacked)
        assert attacked.shape == generate_image.shape
        assert attacked.dtype == generate_image.dtype

    def test_salt_and_pepper_attack_parameters(self, generate_grayscale_image):
        """Test salt and pepper attack with different parameters"""
        # Test different amounts
        for amount in [0.005, 0.01, 0.05]:
            attacked = robustness.salt_and_pepper_attack(generate_grayscale_image, amount=amount)
            assert attacked.shape == generate_grayscale_image.shape
        
        # Test different salt vs pepper ratios
        for ratio in [0.0, 0.5, 1.0]:
            attacked = robustness.salt_and_pepper_attack(
                generate_grayscale_image, amount=0.01, salt_vs_pepper=ratio)
            assert attacked.shape == generate_grayscale_image.shape

    def test_salt_and_pepper_extreme_values(self, generate_grayscale_image):
        """Test that salt and pepper produces extreme values"""
        attacked = robustness.salt_and_pepper_attack(generate_grayscale_image, amount=0.1)
        
        # Should contain some max values (salt)
        assert np.any(attacked == 255)
        # Should contain some min values (pepper)  
        assert np.any(attacked == 0)

    @pytest.mark.skipif(not hasattr(cv2, 'imencode'), reason="OpenCV not properly installed")
    def test_jpeg_compression_attack_basic(self, generate_image):
        """Test JPEG compression attack"""
        # Convert to BGR for OpenCV
        bgr_image = cv2.cvtColor(generate_image, cv2.COLOR_RGB2BGR) if generate_image.ndim == 3 else generate_image
        
        compressed = robustness.jpeg_compression_attack(bgr_image, quality=50)
        
        assert compressed.shape == bgr_image.shape
        # Compression should introduce some changes
        assert not np.array_equal(bgr_image, compressed)

    @pytest.mark.skipif(not hasattr(cv2, 'imencode'), reason="OpenCV not properly installed")
    def test_jpeg_compression_different_qualities(self, generate_grayscale_image):
        """Test JPEG compression with different quality levels"""
        qualities = [10, 50, 90]
        results = []
        
        for quality in qualities:
            compressed = robustness.jpeg_compression_attack(generate_grayscale_image, quality)
            results.append(compressed)
        
        # Higher quality should be closer to original
        mse_original = [np.mean((generate_grayscale_image - result)**2) for result in results]
        assert mse_original[0] > mse_original[1] > mse_original[2]

    def test_cropping_attack_basic(self, generate_image):
        """Test cropping attack"""
        cropped = robustness.cropping_attack(generate_image, crop_fraction=0.1)
        
        assert cropped.shape == generate_image.shape
        # Should be different due to cropping and resizing
        assert not np.array_equal(generate_image, cropped)

    def test_cropping_attack_different_fractions(self, generate_grayscale_image):
        """Test cropping attack with different crop fractions"""
        fractions = [0.05, 0.1, 0.2]
        
        for fraction in fractions:
            cropped = robustness.cropping_attack(generate_grayscale_image, crop_fraction=fraction)
            assert cropped.shape == generate_grayscale_image.shape