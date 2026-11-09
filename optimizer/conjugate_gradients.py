'''
Script para implementação de métodos de 
gradientes conjugados
'''
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from numpy.linalg import cholesky

from contracts.base import Optimizer


class ConjugateGradients(Optimizer):
    '''Classe concreta para implementação do método de
    Gradientes Conjugados'''

    def __init__(self,
                 max_iterations: int = 50,
                 tolerance: float = 10**-2,
                 algorithm: Literal['default', 'conditioned'] = 'default'):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.algorithm = algorithm


    def optimize(self,
                 A: NDArray,
                 b: NDArray,
                 x: NDArray,
                 epsilon: float):
        if self.algorithm == 'default':
            return self._default(A, b, epsilon)
        return self._conditioned(A, x, b, epsilon)


    def _default(A: NDArray, b: NDArray, epsilon: float) -> NDArray:
        '''
        Método iterativo para a solução de sistemas lineares Ax = B, com A sendo
        simétrica e definida positiva
        INPUTS:
        A:matriz dos coeficientes de ordem m x n (array)
        B:vetor de termos independentes m x 1 (array)
        epsilon:tolerancia (escalar)
        OUTPUTS:
        x*:vetor solucao aproximada n x 1 (array) 
        '''
        
        #Inicialização
        i = 0
        x = np.zeros(A.shape[1])
        r = b - A.dot(x)
        r_anterior = r
        
        while np.sqrt(np.dot(r, r)) > epsilon:
            i += 1
            if i == 1:
                p = r
            else:
                beta = np.dot(r, r)/np.dot(r_anterior, r_anterior)
                p = r + beta * p
            
            lamb = np.dot(r, r)/np.dot(p, A.dot(p))
            x += lamb * p
            r_anterior = r
            r += -lamb * A.dot(p)
            
        return x
    

    def _conditioned(self,
                     A: NDArray,
                     x: NDArray,
                     b: NDArray) -> NDArray:
        '''
        Método iterativo para a solução de sistemas lineares Ax = B, com A sendo
        simétrica e definida positiva
        INPUTS:
        A:matriz dos coeficientes de ordem m x n (array)
        B:vetor de termos independentes m x 1 (array)
        x:aproximação inicial n x 1 (escalar)
        OUTPUTS:
        x*:vetor solucao aproximada n x 1 (array)
        '''
        residual = b - A.dot(x)
        preconditioner = np.linalg.inv(cholesky(A))
        
        z = np.dot(preconditioner, residual)
        d = z
        
        error = np.dot(residual.T, residual)
        
        iteration = 0
        while iteration < self.max_iterations and error > self.tolerance**2:
            q        = np.dot( A, d )
            a        = np.dot(residual.T, z)/np.dot( d.T, q )
            
            phi      = np.dot( z.T,  residual )
            old_res  = residual
            
            x        = x + a * d
            residual = residual - a * q
            
            z        = np.dot( preconditioner, residual )
            beta     = np.dot(z.T, (residual-old_res))/phi # Polak-Ribiere
            d        = z + beta * d
            
            error    = residual.T.dot(residual)
            
            iteration += 1


        if iteration < self.max_iterations:
            print('Precisão alcançada. Iterações:', iteration)
        else:
            print('Convergência não alcançada.')
            
        return x

