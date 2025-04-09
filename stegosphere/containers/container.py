from abc import ABC, abstractmethod
import os


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

                

class ContainerCollection(Container):
    """
    Simplifies working with several containers at the same time.
    
    """
    def __init__(self, paths, container_type=None, filter=True):
        self.paths = []
        self.containers = []
        self.data = []
        if isinstance(paths, str):
            if os.path.isdir(paths):
                paths = os.listdir(paths)

        if isinstance(paths, list):
            if container_type == None:
                raise Exception('Unknown container type')
            elif issubclass(container_type, Container):
                for path in paths:
                    try:
                        self.containers.append(container_type(path))
                        self.paths.append(path)
                    except Exception as exc:
                        if not filter:
                            raise exc
    def read(self):
        for container in self.containers:
            self.data.append(container.read())
        return self.data

    def flush(self, data):
        for datum, container in zip(data, self.containers):
            container.flush(datum)

    def save(self, paths):
        for path, container in zip(paths, self.containers):
            container.save(path)

