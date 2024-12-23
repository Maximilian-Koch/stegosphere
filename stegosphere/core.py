from abc import ABC, abstractmethod
import numpy as np
import os


class TransformMethod(ABC):
    """
    A superclass for handling transform-based methods.
    """
    @abstractmethod
    def transform(self, **args):
        pass
    @abstractmethod
    def inverse(self, **args):
        pass


class StegMethod(ABC):
    """
    A superclass for handling steganographic methods.
    """
    @abstractmethod
    def encode(self, **args):
        pass
    @abstractmethod
    def decode(self, **args):
        pass

    
class Container(ABC):
    """
    A superclass for handling files.

    :param file: The file data, either a NumPy array, file path, or list/tuple.
    :type file: numpy.ndarray, str, list, or tuple
    """
    def __init__(self, file):
        self._shape = None
        self.data = self.read(file)
        if hasattr(self.data, 'shape'):
            self._shape = self.data.shape
    
    def read(self, data):
        """
        Reads and returns data from various sources such as NumPy arrays, file paths, or lists/tuples.

        :param data: Data source, either a file path, NumPy array, or list/tuple.
        :type data: str, numpy.ndarray, list, or tuple
        :return: Data as a NumPy array.
        :rtype: numpy.ndarray
        """
        if isinstance(data, np.ndarray):
            return data
        elif os.path.isfile(data):
            return self._read_file(data)  # Specific file reading logic in subclass
        elif isinstance(data, (list, tuple)):
            return np.array(data)
        else:
            raise Exception('format not supported')

    def flush(self):
        """
        Reshapes the file data back to its original shape.
        Use if self.save is not sufficient for saving the result.
        """
        if self._shape is not None:
            self.data = self.data.reshape(self._shape)
        self._flush_file()

    def save(self, path):
        """
        Saves the current data to the specified file path.

        :param path: File path where the data will be saved.
        :type path: str
        """
        self.flush()
        self._save_file(path)  # Specific file saving logic in subclass

    @abstractmethod
    def _flush_file(self):
        """
        Abstract method to be implemented in subclasses for flushing specific file types.
        """
        pass

    @abstractmethod
    def _read_file(self, path):
        """
        Abstract method to be implemented in subclasses for reading specific file types.

        :param path: The file path to read from.
        :type path: str
        """
        raise NotImplementedError("Subclasses must implement `_read_file` method")

    @abstractmethod
    def _save_file(self, path):
        """
        Abstract method to be implemented in subclasses for saving specific file types.

        :param path: The file path to save to.
        :type path: str
        """
        raise NotImplementedError("Subclasses must implement `_save_file` method")
