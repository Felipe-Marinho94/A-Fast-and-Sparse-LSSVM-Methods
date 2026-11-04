'''Script para a decomposição de Cholesky e suas variantes'''

import numpy as np
import math
from numpy.typing import NDArray

from contracts.base import Factorizer

#Decomposição de Cholesky imcompleta
class ImcompleteCholesky(Factorizer):
    '''Classe concreta para implementação da decomposição
    de Cholesky Incompleta'''

    def __init__(self, tikhonov_regularizer: float):
        self.tikhonov_regularizer = tikhonov_regularizer


    def factorize(self, matrix: NDArray) -> NDArray:
        mat = np.copy(matrix) + self.tikhonov_regularizer * np.diag(np.full(matrix.shape[0], 1))
        n = mat.shape[1]
        
        for k in range(n):
            mat[k,k] = math.sqrt( mat[k,k] )
            for i in range(k+1, n):
                if mat[i,k] != 0:
                    mat[i,k] = mat[i,k] / mat[k,k]
            for j in range(k+1, n):
                for i in range(j, n):
                    if mat[i,j] != 0:
                        mat[i,j] = mat[i,j] - mat[i,k] * mat[j,k]
        for i in range(n):
            for j in range(i+1, n):
                mat[i,j] = 0
        
        return mat