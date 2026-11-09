'''
Implmentação de classe para o ajuste (fit) e predict da
proposta LSSVM_ADMM
'''

import numpy as np
from numpy import linalg
from numpy.typing import NDArray

from contracts.base import Model
from kernel.registry import build_kernel
from optimizer.alternating_multipliers import AlternatingDirectionsMultipliers

#------------------------------------------------------------------------------
#Implementando a classe
#------------------------------------------------------------------------------
class PrimalLassoClassifier(Model):
    '''Classe para implementação do classificador LSSVM regularizado
    via LASSO no primal re solvido via ADMM'''
    def __init__(self,
                 tikhonov_regulraization: float = 0.2,
                 kernel: str = 'rbf',
                 **kernel_params)-> None:
        self.tikhonov_regularization = tikhonov_regulraization
        self.kernel = build_kernel(kernel=kernel, kernel_params=kernel_params)

    #------------------------------------------------------------------------------
    #Implementação do método Fit() para a primeira proposta
    #baseada na esparsificação do LSSVM utilizando regularização L1 (LASSO)
    #no problema primal e resolução do problema de otimização pela aplicação
    #do algoritmo Alternating Directions Multipliers Method (ADMM)
    ##Solução aproximada de problemas do tipo min ||b - Ax||^(2) + lambda*||alpha||_{1}
    #------------------------------------------------------------------------------
    def fit(self, X, y):
        '''
        Interface do método
        Ação: Es te método visa realizar o ajuste do modelo LSSVM com regularização L1
        via LASSO no problema primal e resolvido pelo ADMM.

        INPUT
            X: Matriz de Dados (array n x p)
            y: Vetor de rótulos (classificação) 
               ou vetor de respostas numéricas (regressão) (array n x 1)
            kernel: kernel utlizado ("linear", "polinomial", "gaussiano") (string)
            tau: Termo de regularização do problema primal do LSSVM (escalar)
        
        OUTPUT
            x_ótimo: Vetor solução aproximado do sistema KKT Ax = b (array n+1 x 1)
            x_ótimo = [alphas b].T
            b = 0
        '''

        # Construindo a matriz de kernel
        n_samples, _ = X.shape
        K = self.kernel.compute(X)
        
        #Cholesky Incompleta
        P = np.linalg.cholesky(K + self.tikhonov_regularization * np.diag(np.full(K.shape[0], 1)))
        
        #Construção da matriz dos coeficiente A
        A = P.T
        
        #Construção do vetor de coeficientes b
        b = np.dot(linalg.inv(self.tikhonov_regularization * np.identity(n_samples) + np.dot(P, P.T)), np.dot(P.T, y))
        b = np.expand_dims(b, axis = 1)
        
        #Solução do sistema KKT em conjunto com o LASSO
        optimizer = AlternatingDirectionsMultipliers(algorithm='ADMM', l=0.01, rho=1.0)
        solution, _, _  = optimizer.optimize(A, y=b)

        #Obtenção dos multiplicadores de Lagrange
        alphas = solution
        idx_continua = np.where(alphas != 0)[0]

        self.X_train = X
        self.alphas = alphas

        resultado = {"mult_lagrange": alphas,
                    "kernel": K,
                    "A": A,
                    "b": 0,
                    "sv_x": X[idx_continua, :]}
        
        return resultado
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para a primeira proposta considerando um
    #problema de classificação
    #------------------------------------------------------------------------------
    def predict(self,
                X_teste: NDArray[np.float64])-> NDArray[np.float64]:
        
            return self._predict_with(self.alphas, self.X_train, X_teste)
    
    def _predict_with(self,
                      alphas: NDArray,
                      X_treino: NDArray,
                      X_teste: NDArray)-> NDArray:
            '''Método auxiliar para predição com alphas específicos'''
            return np.sign(self.kernel.compute(X_teste, X_treino) @ alphas)

    