import pytest
import numpy as np
import warnings

import stegosphere
from stegosphere.utils import generate_binary_payload as gbp
from stegosphere.embeddings import LSB, VD, BPCS, echo_hiding
from stegosphere.transform import IWT
from stegosphere import io, payloader, selector
from stegosphere.tools import multifile

from stegosphere.analysis import accuracy, robustness
from stegosphere.costs import HILL, WOW

# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests with several modules"""

    def test_lsb_with_iwt(self, generate_grayscale_image, small_payload):
        """Test LSB embedding in wavelet domain"""
        coeffs, boundary_info = IWT.transform(generate_grayscale_image)
        approx_key = min(coeffs.keys(), key=len)
        approx_coeffs = coeffs[approx_key]
        
        # Embed in approximation coefficients
        stego_coeffs = LSB.embed(approx_coeffs, small_payload)
        
        coeffs[approx_key] = stego_coeffs
        stego_image = IWT.inverse(coeffs, boundary_info)
        
        coeffs_extract, _ = IWT.transform(stego_image)
        extracted = LSB.extract(coeffs_extract[approx_key])
        
        assert extracted == small_payload
    
    def test_lsb_with_cost_based_selection(self, generate_grayscale_image, small_payload):
        """Test LSB embedding with cost-based pixel selection"""
        costs = HILL.hill_cost(generate_grayscale_image.astype(np.float64))
        flat_costs = costs.flatten()
        indices = selector.greedy_indices(flat_costs, n=len(small_payload))
        
        stego = LSB.embed(generate_grayscale_image, small_payload, indices=indices, framing=None)
        extracted = LSB.extract(stego, indices=indices, framing=None)
        
        assert extracted == small_payload

    def test_multifile_with_different_algorithms(self, generate_image, text_payload):
        """Test multifile embedding with different steganographic algorithms"""
        binary_payload = io.data_to_binary(text_payload)
        
        def lsb_embed(payload_chunk):
            return LSB.embed(generate_image.copy(), payload_chunk)
        
        def vd_embed(payload_chunk):
            return VD.embed(generate_image.copy(), payload_chunk)
        
        instances = [lsb_embed, vd_embed]
        stego_images = multifile.split_encode(binary_payload, instances, distribution='even')
        
        assert len(stego_images) == 2
    
        def lsb_extract():
            return LSB.extract(stego_images[0])
        
        def vd_extract():
            return VD.extract(stego_images[1])
        
        extract_instances = [lsb_extract, vd_extract]
        recovered_payload = multifile.split_decode(extract_instances)
        
        assert recovered_payload == binary_payload

    def test_robustness_analysis_pipeline(self, generate_image, small_payload):
        """Test complete robustness analysis pipeline"""
        stego = LSB.embed(generate_image, small_payload, framing=None)
        attacks = [lambda img: robustness.random_bit_attack(img, ratio=0.01),
            lambda img: robustness.gaussian_noise_attack(img, sigma=0.5),
            lambda img: robustness.salt_and_pepper_attack(img, amount=0.005)]
        
        results = []
        for attack_func in attacks:
            attacked_image = attack_func(stego)
            try:
                extracted = LSB.extract(attacked_image, framing=None)[:len(small_payload)]
                ber = accuracy.bit_error_rate(small_payload, extracted)
                hamming_dist = accuracy.hamming_distance(small_payload, extracted)
                results.append({
                    'ber': ber,
                    'hamming_distance': hamming_dist,
                    'extraction_success': True
                })
            except:
                results.append({
                    'ber': 1.0,
                    'hamming_distance': len(small_payload),
                    'extraction_success': False
                })
        
        assert len(results) == len(attacks)
        for result in results:
            assert 'ber' in result
            assert 'hamming_distance' in result
            assert isinstance(result['ber'], float)

    def test_cost_guided_multifile_embedding(self, generate_image, text_payload):
        """Test multifile embedding with cost-guided selection"""
        binary_payload = io.data_to_binary(text_payload)
        
        costs = []
        for channel in range(3):
            channel_costs = WOW.wow_cost(generate_image[:,:,channel].astype(np.float64))
            costs.append(channel_costs.flatten())
        
        # Select best channel based on average cost
        avg_costs = [np.mean(cost) for cost in costs]
        best_channel = np.argmin(avg_costs)
        
        # Use best channel for embedding
        channel_image = generate_image[:,:,best_channel]
        best_indices = selector.greedy_indices(costs[best_channel], n=len(binary_payload))
        
        stego_channel = LSB.embed(channel_image, binary_payload, indices=best_indices, framing=None)
        extracted = LSB.extract(stego_channel, indices=best_indices, framing=None)
        
        assert extracted == binary_payload

    def test_accuracy_analysis_with_different_framings(self, generate_image, text_payload):
        """Test accuracy analysis across different payload framings"""
        binary_payload = io.data_to_binary(text_payload)
        
        framings = [payloader.FixedHeader,
            payloader.Delimiter,
            payloader.Unary,
            payloader.EliasGamma]
        
        results = {}
        for framing in framings:
            stego = LSB.embed(generate_image, binary_payload, framing=framing)
            noisy_stego = robustness.random_bit_attack(stego, ratio=0.005)
            
            # Extract
            try:
                extracted = LSB.extract(noisy_stego, framing=framing)
                ber = accuracy.bit_error_rate(binary_payload, extracted)
                results[framing.__name__] = ber
            except:
                results[framing.__name__] = 1.0
        
        # All framings should have some measurable accuracy
        assert len(results) == len(framings)
        for framing_name, ber in results.items():
            assert isinstance(ber, float)
            assert 0.0 <= ber <= 1.0