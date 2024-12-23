import warnings
try:
    from PIL import Image
except:
    warnings.warn("Importing pillow failed. pillow is used for reading/writing images. Images will need to be loaded into numpy array format otherwise.",
                  UserWarning)
import numpy as np

import core
from utils import *
import utils
from spatial import BaseLSB, BaseVD
from transform import BaseIWT

__all__ = ['IWT', 'LSB', 'PVD', 'METADATA_LENGTH_IMAGE']

class ImageHandler(core.Container):
    """
    A base class for handling image data.

    :param image: The image data, which can be a NumPy array, file path, or list/tuple.
    :type image: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, image):
        super().__init__(image)
    def _read_file(self, path):
        self.obj = Image.open(path)
        return np.array(self.obj)
    def _flush_file(self):
        if len(self._shape) <= 2: #grayscale
            data = self.data.flatten()
        else: #multiple channels
            data = [tuple(pixel) for row in self.data for pixel in row]
        self.obj.putdata(data)
    def _save_file(self, path):
        self.obj.save(path)

class LSB(ImageHandler, BaseLSB):
    """
    Implements LSB (Least Significant Bit) steganography for hiding messages in images.

    :param image: The image data, either a NumPy array, file path, or list/tuple.
    :type image: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, image, analysis=utils.ANALYSIS):
        ImageHandler.__init__(self, image)
        BaseLSB.__init__(self, self.data, analysis)

class PVD(ImageHandler, BaseVD):
    """
    Implements PVD (Pixel Value Differencing) steganography for hiding messages in images.

    :param image: The image data, either a NumPy array, file path, or list/tuple.
    :type image: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, image, analysis=utils.ANALYSIS):
        ImageHandler.__init__(self, image)
        if len(self._shape) == 3:
            pos_dim = 2
            depth = self._shape[2]
        elif len(self._shape) == 2:
            pos_dim = 2
            depth = 1
        else:
            #this library does not support voxels (yet)
            raise Exception('unsupported image shape')
        BaseVD.__init__(self, self.data, pos_dim, depth, analysis)


class IWT(ImageHandler, BaseIWT):
    def __init__(self, image):
        ImageHandler.__init__(self, image)
        BaseIWT.__init__(self, self.data)
