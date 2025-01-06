from abc import ABC, abstractmethod

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
    def embed(self, **args):
        pass
    @abstractmethod
    def extract(self, **args):
        pass


class Container(ABC):
    """
    A superclass for file containers.
    """
    @abstractmethod
    def read(self, **args):
        pass
    @abstractmethod
    def flush(self, **args):
        pass
    @abstractmethod
    def save(self, **args):
        pass
    
