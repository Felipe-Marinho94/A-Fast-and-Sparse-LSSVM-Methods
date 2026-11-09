'''Script com a s funções utilitárias para o modelo
LSSVM_ADMM'''
from typing import Any, Literal, Optional

import numpy as np
from numpy.typing import NDArray
from math import sqrt, log

from contracts.base import Optimizer


class AlternatingDirectionsMultipliers(Optimizer):
    '''classe para implementação '''

    def __init__(self,
                 algorithm: Literal['ADMM', 'Fast_ADMM', 'Fast_ADMM_restart'] = 'ADMM',
                 l: Optional[float] = None,
                 rho: Optional[float] = None) -> None:
        self.algorithm = algorithm
        self.l = l
        self.rho = rho
        

    def _Sthresh(self, x: NDArray, gamma: float) -> float:
        return np.sign(x)*np.maximum(0, np.absolute(x)-gamma/2.0)

    
    def ADMM(self,
             A: NDArray,
             y: NDArray,
             max_itetarions: int = 10000) -> tuple[Any]:

        _, n = A.shape
        w, _ = np.linalg.eig(A.T.dot(A))

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        xhat = np.zeros([n, 1])
        zhat = np.zeros([n, 1])
        u = np.zeros([n, 1])

        "Calcular os coeficientes e tamanho do passo"
        l   = self.l   if self.l   is not None else sqrt(2*log(n, 10))
        rho = self.rho if self.rho is not None else 1/(np.amax(np.absolute(w)))

        "Pre-cácula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < max_itetarions):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - u))

            "z minimização via soft-thresholding"
            zhat = self._Sthresh(xhat + u, l/rho)

            "atualização dos multiplicadores"
            u = u + xhat - zhat

            i = i+1
        return zhat, rho, l

    
    def Fast_ADMM(self,
                  A: NDArray,
                  y: NDArray,
                  l: float,
                  rho:float,
                  max_iterations: int = 100000) -> tuple[Any]:

        _, n = A.shape

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        #Inicialização
        xhat = np.zeros([n, 1])
            
        z = np.zeros([n, 1])
        zhat = np.zeros([n, 1])

        u = np.zeros([n, 1])
        uhat = np.zeros([n, 1])

        alpha = 1 #termo de momento

        "Pre-cálcula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < max_iterations):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - uhat))

            "z minimização via soft-thresholding"
            z_anterior = z
            z = self._Sthresh(xhat + uhat, l/rho)

            "atualização dos multiplicadores"
            u_anterior = u
            u = uhat + xhat - z

            "atualização do termo de momento"
            alpha_anterior = alpha
            alpha = (1+sqrt(1+4*(alpha**2)))/2

            "Atualização do vetor z"
            zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

            "Atualização dos multiplicadores de Lagrange"
            uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

            i += 1
        return zhat, rho, l

    
    def Fast_ADMM_restart(self,
                          A: NDArray,
                          y: NDArray,
                          tau: float,
                          neta: float,
                          max_iterations: int = 10000) -> tuple[Any]:

        _, n = A.shape
        w, _ = np.linalg.eig(A.T.dot(A))

        "Função para calcular min 1/2(y - Ax) + l||x||"
        "via alternating direction methods"
        #Inicialização
        xhat = np.zeros([n, 1])
            
        z = np.zeros([n, 1])
        zhat = np.zeros([n, 1])

        u = np.zeros([n, 1])
        uhat = np.zeros([n, 1])

        alpha = 1 #termo de momento
        c = 0

        "Calcular os coeficientes e tamanho do passo"
        l = sqrt(2*log(n, 10))
        rho = 1/(np.amax(np.absolute(w)))

        "Pre-cálcula para salva tais operações"
        AtA = A.T.dot(A)
        Aty = A.T.dot(y)
        Q = AtA + rho*np.identity(n)
        Q = np.linalg.inv(Q)

        i = 0

        while(i < max_iterations):

            "x minimização via OLS"
            xhat = Q.dot(Aty + rho*(zhat - uhat))

            "z minimização via soft-thresholding"
            z_anterior = z
            z =self._Sthresh(xhat + uhat, l/rho)

            "atualização dos multiplicadores"
            u_anterior = u
            u = uhat + xhat - z

            #Regra para reinicialização
            c_anterior = c
            c = (1/tau)*(np.linalg.norm(u - uhat)**2) + tau*(np.linalg.norm(z - zhat)**2)

            alpha_anterior = alpha
            if c < neta*c_anterior:
                "atualização do termo de momento"
                alpha = (1+sqrt(1+(4*alpha**2)))/2

                "Atualização do vetor z"
                zhat = z + (alpha_anterior -1)*(z-z_anterior)/alpha

                "Atualização dos multiplicadores de Lagrange"
                uhat = u + (alpha_anterior -1)*(u-u_anterior)/alpha

            else:
                alpha = 1
                zhat = z_anterior
                uhat = u_anterior
                c = c/neta
            i += 1
            return zhat, rho, l


    def optimize(self, A: NDArray, y: NDArray, **kwargs: Any) -> tuple[Any]:
        '''função que seleciona o algoritmo a ser utilizado'''
        func = getattr(self, self.algorithm)
        return func(A, y, **kwargs)
