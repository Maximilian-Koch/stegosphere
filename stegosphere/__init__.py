"""
Stegosphere is a versatile library for both
applying steganography (hiding data within media) and
performing steganalysis (detecting hidden data).
It provides a flexible framework for any data representable
as a NumPy array, including images, videos, audios and TrueType fonts.
Furthermore, multi-file embedding, Hamming codes, payload compression, encryption and visualization are available, as well as a research toolbox for evaluating method performance and security.
"""

from stegosphere.io import binary_to_data, binary_to_file, data_to_binary, file_to_binary
from stegosphere.containers import image, video, audio, font
from stegosphere import embeddings, transform, costs, analysis, selector

__all__ = ['binary_to_data', 'binary_to_file', 'data_to_binary', 'file_to_binary',
            'image','audio','video','font', 'embeddings', 'costs', 'transform',
             'analysis', 'selector']

__author__ = 'Maximilian J. W. Koch'

__version__ = '1.3'

def demo():
    """
    Runs demo by extracting a payload from the logo!
    """
    img = image.ImageContainer('logo_small_stego.png')
    px = img.read()
    payload = embeddings.LSB.extract(px)
    readable_payload = binary_to_data(payload).decode()
    if 'stegosphere' in readable_payload:
        print(readable_payload)
    else:
        print('Decoding the message did not succeed.')


if __name__ == '__main__':
    demo()