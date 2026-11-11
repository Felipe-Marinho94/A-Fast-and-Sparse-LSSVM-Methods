'''Script para implementação abstrações comuns
a todos os modelos'''

from typing import Dict, Optional

import numpy as np
from numpy.typing import NDArray
from numpy import linalg
from pydantic import BaseModel


class BuildKKTSystem(BaseModel):
    '''Classe construtora de sistemas KKT'''
    kernel_matrix: NDArray
    tikhonov_regularization: float
    primal_regularization: Optional[float] = None

    def build_classification_without_bias(self,
                            X: NDArray,
                            y: NDArray)-> Dict[str, NDArray]:
        '''Método para construção do sistema KKT
        no problema de treinamento de LSSVM's para tarefas de classficação
        sem um termo de bias'''
        
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
    

    def build_classification(self,
                             X: NDArray,
                             y: NDArray)-> Dict[str, NDArray]:
        '''Método para construção do sistema KKT
        no problema de treinamento de LSSVM's para tarefas de classficação'''

        # Construção da matriz dos coeficientes
        n_samples = X.shape[0]

        #Construção da Matriz Omega
        Omega = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                Omega[i, j] = y[i] * y[j] * self.kernel_matrix[i, j]
                
        #Construção da Matriz A
        H = Omega + (1/self.tikhonov_regularization) * np.identity(n_samples)
        A = np.block([[np.array([0]), np.expand_dims(y, axis = 1).T],
                      [np.expand_dims(y, axis = 1), H]])
        
        #Construção do Vetor B
        B = np.concatenate((np.expand_dims(np.zeros([1]), axis=1),
                            np.expand_dims(np.ones(n_samples), axis = 1)), axis=0)
                
        return {'A': A, 'B' : B}
            
    
    def build_regression(self,
                         X: NDArray,
                         y: NDArray)-> Dict[str, NDArray]:
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


    
