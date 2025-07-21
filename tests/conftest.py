import pytest
import numpy as np
from stegosphere.utils import generate_binary_payload as gbp

@pytest.fixture
def generate_image():
    """Generate a test RGB image"""
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

@pytest.fixture 
def generate_grayscale_image():
    """Generate a test grayscale image"""
    return np.random.randint(0, 256, (50, 50), dtype=np.uint8)

@pytest.fixture
def generate_audio():
    """Generate test audio signal"""
    return np.random.randn(44100).astype(np.float32)

@pytest.fixture
def generate_stereo_audio():
    """Generate test stereo audio signal"""
    return np.random.randn(22050, 2).astype(np.float32)

@pytest.fixture
def payload_generator():
    """Generate binary payload"""
    return gbp(1000)

@pytest.fixture
def small_payload():
    """Generate small binary payload"""
    return gbp(100)

@pytest.fixture
def text_payload():
    """Generate text payload"""
    return "Hello, World! This is a stegosphere test payload."

@pytest.fixture
def binary_payload():
    """Generate pure binary string payload"""
    return "11010110100100111001001100100010001110010101"

@pytest.fixture
def hex_payload():
    """Generate hexadecimal payload"""
    return "78fc190a1757b284e015298ab3"