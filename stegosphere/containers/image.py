import os
import numpy as np
try:
    import PIL.Image
except:
    raise Exception('ImageContainer requires PIL. Use array container instead.')

from stegosphere import base

class ImageContainer(base.Container):
    """
    File container for image files.
    """
    def __init__(self, image):
        if isinstance(image, PIL.Image.Image):
            self.image = image
        elif os.path.isfile(image):
            self.image = PIL.Image.open(image)
        elif isinstance(image, np.ndarray):
            raise Exception('No need for containers for arrays')
        else:
            raise Exception('image must be PIL.Image or path')
        
    def read(self):
        """
        Read pixels
        """
        return np.array(self.image)
    def flush(self, pixels):
        """
        Flush pixels into container
        """
        assert isinstance(pixels, np.ndarray)
        if len(pixels.shape) <= 2: #grayscale
            data = pixels.flatten()
        else: #multiple channels
            data = [tuple(pixel) for row in pixels for pixel in row]
        self.image.putdata(data)
    def save(self, path):
        """
        Save image to path
        """
        self.image.save(path)
            
