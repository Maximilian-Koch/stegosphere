import pytest
import numpy as np

from stegosphere.costs import HILL, WOW

# =============================================================================
# COST FUNCTION TESTS
# =============================================================================

class TestCostFunctions:
    """Test cases for cost function modules"""
    
    def test_hill_cost_basic(self, generate_grayscale_image):
        """Test basic HILL cost computation"""
        costs = HILL.hill_cost(generate_grayscale_image.astype(np.float64))
        
        assert costs.shape == generate_grayscale_image.shape
        assert np.all(costs > 0)  # All costs should be positive
        assert np.all(np.isfinite(costs))  # No infinite values

    def test_hill_cost_different_alphas(self, generate_grayscale_image):
        """Test HILL cost with different alpha values"""
        alphas = [0.1, 1.0, 10.0]
        results = []
        
        for alpha in alphas:
            costs = HILL.hill_cost(generate_grayscale_image.astype(np.float64), alpha=alpha)
            results.append(costs)
            assert costs.shape == generate_grayscale_image.shape
            assert np.all(costs > 0)

    def test_hill_cost_different_sizes(self, generate_grayscale_image):
        """Test HILL cost with different kernel sizes"""
        sizes = [3, 5, 7]
        
        for size in sizes:
            costs = HILL.hill_cost(generate_grayscale_image.astype(np.float64), size=size)
            assert costs.shape == generate_grayscale_image.shape
            assert np.all(costs > 0)

    def test_hill_cost_edge_response(self):
        """Test that HILL cost responds appropriately to edges"""
        # Create image with clear edge
        smooth_region = np.ones((50, 50))
        edge_region = np.zeros((50, 50))
        edge_region[25:, :] = 255  # Horizontal edge
        
        smooth_costs = HILL.hill_cost(smooth_region)
        edge_costs = HILL.hill_cost(edge_region)
        
        # Edge region should have lower costs (higher embedding capacity) 
        # in areas away from the edge
        assert np.mean(edge_costs) < np.mean(smooth_costs)

    def test_wow_cost_basic(self, generate_grayscale_image):
        """Test basic WOW cost computation"""
        costs = WOW.wow_cost(generate_grayscale_image.astype(np.float64))
        
        assert costs.shape == generate_grayscale_image.shape
        assert np.all(costs > 0)  # All costs should be positive
        assert np.all(np.isfinite(costs))  # No infinite values

    def test_wow_cost_different_alphas(self, generate_grayscale_image):
        """Test WOW cost with different alpha values"""
        alphas = [1.0, 10.0, 100.0]
        results = []
        
        for alpha in alphas:
            costs = WOW.wow_cost(generate_grayscale_image.astype(np.float64), alpha=alpha)
            results.append(costs)
            assert costs.shape == generate_grayscale_image.shape
            assert np.all(costs > 0)

    def test_wow_cost_texture_response(self):
        """Test that WOW cost responds to texture appropriately"""
        # Create smooth and textured regions
        smooth = np.ones((50, 50)) * 128
        
        # Create checkerboard texture
        textured = np.zeros((50, 50))
        textured[::2, ::2] = 255
        textured[1::2, 1::2] = 255
        
        smooth_costs = WOW.wow_cost(smooth)
        textured_costs = WOW.wow_cost(textured)
        
        # Textured regions should generally have lower costs 
        # (better for embedding)
        assert np.mean(textured_costs) < np.mean(smooth_costs)

    def test_cost_functions_color_images(self, generate_image):
        """Test cost functions with color images (should work on each channel)"""
        # Test HILL on color image
        hill_costs = []
        for channel in range(3):
            costs = HILL.hill_cost(generate_image[:,:,channel].astype(np.float64))
            hill_costs.append(costs)
            assert costs.shape == generate_image[:,:,channel].shape
        
        # Test WOW on color image
        wow_costs = []
        for channel in range(3):
            costs = WOW.wow_cost(generate_image[:,:,channel].astype(np.float64))
            wow_costs.append(costs)
            assert costs.shape == generate_image[:,:,channel].shape

    def test_cost_functions_with_convolution_modes(self, generate_grayscale_image):
        """Test that cost functions handle convolution edge cases"""
        # Small image to test edge handling
        small_image = generate_grayscale_image[:10, :10].astype(np.float64)
        
        hill_costs = HILL.hill_cost(small_image)
        wow_costs = WOW.wow_cost(small_image)
        
        assert hill_costs.shape == small_image.shape
        assert wow_costs.shape == small_image.shape
        assert np.all(np.isfinite(hill_costs))
        assert np.all(np.isfinite(wow_costs))
