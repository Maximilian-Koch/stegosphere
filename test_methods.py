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


def test_method_accuracy(generate_image, payload_generator):
    methods = [LSB, VD, BPCS]
    for method in methods:
        image = generate_image
        payload = payload_generator

        embedded = method.embed(image, payload)
        if method == BPCS:
            extracted = method.extract(*embedded)
        else:
            extracted = method.extract(embedded)

        assert payload == extracted
