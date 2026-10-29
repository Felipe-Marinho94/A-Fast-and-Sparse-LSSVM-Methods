'''Script para abstração de kernels'''

from abc import ABC, abstractmethod
import numpy.typing as nptype

class Kernel(ABC):
    '''Abstração para implementação de novos Kernels'''
    @abstractmethod
    def compute(self,
                X: nptype.NDArray,
                Y: nptype.NDArray = None)-> nptype.NDArray:...
    