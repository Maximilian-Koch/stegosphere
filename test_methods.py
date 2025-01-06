import pytest
import numpy as np

import stegosphere

from stegosphere.methods import LSB, VD, IWT, BPCS
from stegosphere.utils import generate_binary_payload as gbp

@pytest.fixture
def generate_image():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)

@pytest.fixture
def payload_generator():
    return gbp(1000)


def test_lsb_accuracy(generate_image, payload_generator):
    image = generate_image
    payload = payload_generator

    embedded = LSB.embed(image, payload)
    extracted = LSB.extract(embedded)

    assert payload == extracted
