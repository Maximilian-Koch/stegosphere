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


