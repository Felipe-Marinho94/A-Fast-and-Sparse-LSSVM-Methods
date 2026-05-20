'''Script para a realização de produtos internos
em kernel gaussianos'''

#-------------------------------------
# Carregando pacotes relevantes
#-------------------------------------
import numpy as np
import numpy.typing as nptype
from kernel.base import Kernel

class RBFKernel(Kernel):
    def __init__(self, sigma: float = 0.5)-> None:
        super().__init__()
        self.sigma = sigma

    def compute(self,
                X: nptype.NDArray, 
                Y: nptype.NDArray = None)-> nptype.NDArray:
        if Y is None:
            Y = X
        
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)

        dist = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-self.sigma * dist) 
