'''Script para implementação do cálculo do 
produto interno no kernel polinomial'''

#-----------------------------------------
# Carregando pacotes relevantes
#-----------------------------------------
from typing import Dict, Optional

import numpy.typing as nptype
from kernel.base import Kernel

class PolynomialKernel(Kernel):
    '''classe concreta para o cálculo de produtos internos
    em kernel polinomial'''
    def __init__(self, kernel_params: Optional[Dict[str, float]] = None)-> None:
        super().__init__()
        kernel_params = kernel_params or {}
        self.C = kernel_params.get('C', 1)
        self.degree = kernel_params.get('degree', 3)

    def compute(self,
                X: nptype.NDArray,
                Y: nptype.NDArray = None)-> nptype.NDArray:
        if Y is None:
            Y = X
        return (X @ Y.T + self.C) ** self.degree
