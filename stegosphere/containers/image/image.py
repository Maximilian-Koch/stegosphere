"""
A module for image containers
"""
import os
import numpy as np
from stegosphere.containers import container

try:
    from PIL import Image
except ImportError as e:
    raise ImportError("ImageContainer requires Pillow (PIL). Install via `pip install pillow`. Alternatively, read the image directly as array.") from e


class ImageContainer(container.Container):
    """
    File container for image files.
    """
    def __init__(self, image):
        if hasattr(Image, 'Image') and isinstance(image, Image.Image):
            self.image = image
        elif isinstance(image, str):
            if not os.path.isfile(image):
                raise FileNotFoundError(f"No such image file: '{image}'")
            try:
                self.image = Image.open(image)
            except (OSError, Image.UnidentifiedImageError) as e:
                raise OSError(f"Cannot open image '{image}': {e}") from e
        else:
            raise TypeError("ImageContainer requires a PIL.Image.Image or valid image file path")

        
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
    def save(self, path, flush=None):
        """
        Save image to path

        Args:
            path (str): Path to save file
            flush (numpy.ndarray | None): Data to flush before saving
        Returns:
            None
        """
        if flush is not None:
            self.flush(flush)
        self.image.save(path)
            
