from abc import ABC, abstractmethod


class BaseAlgorithm(ABC):
    """Base class for all qc2 algos."""

    @abstractmethod
    def run(self, *args, **kwargs):
        """run it"""
