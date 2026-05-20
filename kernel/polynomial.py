'''Script para implementação do cálculo do 
produto interno no kernel polinomial'''

#-----------------------------------------
# Carregando pacotes relevantes
#-----------------------------------------
import numpy.typing as nptype
from kernel.base import Kernel

class PolynomialKernel(Kernel):
    '''classe concreta para o cálculo de produtos internos
    em kernel polinomial'''
    def __init__(self, C: int = 1, degree: int = 3)-> None:
        super().__init__()
        self.C = C
        self.degree = degree

    def compute(self,
                X: nptype.NDArray,
                Y: nptype.NDArray = None)-> nptype.NDArray:
        if Y is None:
            Y = X
        return (X @ Y.T + self.C) ** self.degree
