'''Script para implementação abstrações comuns
a todos os modelos'''

#--------------------------------------------------------
# Carregando pacotes relevantes
#--------------------------------------------------------
from typing import Dict
import numpy as np
import numpy.typing as nptype
from numpy import linalg
from pydantic import BaseModel

#--------------------------------------------------------
# Classe construtora do sistema KKT
#--------------------------------------------------------
class BuildKKTSystem(BaseModel):
    '''Classe construtora de sistemas KKT'''
    kernel_matrix: nptype.NDArray
    tikhonov_regularization: float
    primal_regularization: float

    def build_classfication(self,
                            X: nptype.NDArray,
                            y: nptype.NDArray)-> Dict[str, nptype.NDArray]:
        '''Método para construção do sistema KKT
        no problema de treinamento de LSSVM's para tarefas de classficação'''
        
        # Decomposição da matriz de kernel
        P = linalg.cholesky(self.kernel_matrix + 
                            self.tikhonov_regularization * 
                            np.diag(np.full(self.kernel_matrix.shape[0], 1)))
        
        # Construção da matriz dos coeficientes
        A = P.T

        # Construção do vetor de coeficientes B
        B = np.dot(linalg.inv(self.primal_regularization * np.identity(X.shape[0]) + np.dot(P.T, P)),
                   np.dot(P.T, y))
        
        return {'A': A, 'B': B}
    
    def build_regression(self, X: nptype.NDArray, y: nptype.NDArray)-> Dict[str, nptype.NDArray]:
        '''Método para construção dos sistema KKT no problema de
        treinamento de LSSVM's em tarefas de regressão'''

        # Construção da matriz omega
        omega_matrix = self.kernel_matrix

        # Construção da matriz dos coeficientes
        n_samples = X.shape[0]
        H = omega_matrix + (1/self.primal_regularization) * np.identity(n_samples)
        one_column = np.ones((n_samples))
        s = np.dot(one_column, linalg.inv(H).dot(one_column))
        zero_line, zero_column = np.zeros((1, n_samples)), np.zeros((n_samples, 1))
        A = np.block([[s, zero_line], [zero_column, H]])

        # Construção do vetor B
        d2 = np.expand_dims(y, axis = 1)
        b1 = np.expand_dims(np.array(np.dot(one_column, linalg.inv(H).dot(y))), axis = 0)
        B = np.concatenate((np.expand_dims(b1, axis = 1), d2), axis = 0)
        B = np.squeeze(B)

        return {'A': A, 'B': B}


    
