'''Script para implementação do cálculo de produtos internos
utilizando o kernel linear'''

#-----------------------------------------
# Carregando pacotes relevantes
#-----------------------------------------
import numpy.typing as nptype
from kernel.base import Kernel

class LinearKernel(Kernel):
    '''Classe concreta para o cálculo de
      produtos internos em kernel linear'''
    def compute(self, 
                X: nptype.NDArray,
                Y: nptype.NDArray = None)-> nptype.NDArray:
        if Y is None:
            Y = X
        return X @ Y.T
