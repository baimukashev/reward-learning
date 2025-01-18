from abc import ABCMeta
from abc import abstractmethod


class BaseAlgo(object, metaclass=ABCMeta):
    """Abstract agent class."""

    def __init__(self, cfg) -> None:
        self.cfg = cfg

    @abstractmethod
    def log_data(self, data):
        pass

    @abstractmethod
    def train(self):
        pass
