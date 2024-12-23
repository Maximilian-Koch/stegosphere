import warnings

import numpy as np
try:
    from scipy.io import wavfile
except:
    warnings.warn('Failed importing scipy. wav-file will need to be read/saved manually.')

import os

from utils import *
from spatial import BaseLSB, BaseVD
from transform import BaseIWT
import core

__all__ = ['FVD', 'LSB', 'IWT', 'METADATA_LENGTH_AUDIO']

class Audio(core.Container):
    def __init__(self, audio):
        self.rate = None
        super().__init__(audio)
    def _read_file(self, path):
        self.rate, data = wavfile.read(path)
        return data
    def _flush_file(self):
        pass
    def _save_file(self, path):
        assert self.rate is not None
        wavfile.write(path, self.rate, self.data)

class LSB(Audio, BaseLSB):
    """
    Implements LSB (Least Significant Bit) steganography for hiding messages in audio files.

    :param audio: The audio data, either as a NumPy array, file path, or list/tuple.
    :type audio: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, audio):
        Audio.__init__(self, audio)
        BaseLSB.__init__(self, self.data)

class FVD(Audio, BaseVD):
    """
    Implements FVD (Frequency Value Differencing) steganography for hiding messages in audio files.

    :param audio: The audio data, either as a NumPy array, file path, or list/tuple.
    :type audio: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, audio):
        Audio.__init__(self, audio)
        assert len(self._shape)<=2
        BaseVD.__init__(self, self.data, 1, self._shape[-1])


class IWT(Audio, BaseIWT):
    def __init__(self, image):
        Audio.__init__(self, image)
        BaseIWT.__init__(self, self.data)
