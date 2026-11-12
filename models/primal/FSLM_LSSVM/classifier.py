'''
Implementação da classe para proposta Fixed Sized Levenberg-Marquardt
'''

import numpy as np
from numpy.typing import NDArray
from numpy import linalg

from contracts.base import Model
from kernel.registry import build_kernel
from models.base import BuildKKTSystem

class FixedSizedLMClassifier(Model):
    '''Classe concreta para a implementação do modelo
    Fixed Sized Levenberg-Marquardt LSSVM'''
    def __init__(self,
                     tikhonov_regulraization: float = 0.2,
                     kernel: str = 'rbf',
                     **kernel_params)-> None:
            self.tikhonov_regularization = tikhonov_regulraization
            self.kernel = build_kernel(kernel=kernel, kernel_params=kernel_params)

    #----------------------------------------------------------------
    #Implementação da Proposta 
    # Fixed Sized Levenberg-Marquardt LSSVM (FSLM-LSSVM)
    #Source:https://www.researchgate.net/publication/377073903_New_Iterative_Pruning_Methods_for_Least_Squares_Support_Vectors_Machines
    #----------------------------------------------------------------
    #Implementação do Método Fit() para a proposta FSLM-LSSVM

    def fit(self,
            X: NDArray,
            y: NDArray,
            max_iterations: int = 100,
            window_size: int = 5,
            step_size: float = 0.5,
            tolerance: float = 0.01,
            percentage: float = 0.2):
        """
        Interface do método
        Ação: Este método visa realizar a poda iterativa dos vetores
        de suporte estimados no LSSVM, trabalhando com a matriz dos
        coeficientes completa em todas as iterações.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        step_size: Taxa de aprendizado (escalar);
        window_size: Termo que define a faixa de poda (escalar);
        percentage: Percentual de remoção (escalar);
        max_iterations: Número máximo de iterações (escalar);
        tolerance: Tolerância (Critério de parada).

        OUTPUT:
        vetor esparso de multiplicadores de Lagrange estimados. 
        """

        #Construção da matriz A e vetor b e inicialização aleatória
        #do vetor de multiplicadores de Lagrange
        n_samples, _ = X.shape 
        K = self.kernel.compute(X)
        
        builder = BuildKKTSystem(kernel_matrix=K, tikhonov_regularization=self.tikhonov_regularization)
        A, B = builder.build_classification(X, y)
        
        #Inicialização aleatória do vetor de multiplicadores de Lagrange
        z_inicial = np.random.normal(loc=0, scale=1, size=(n_samples + 1, 1))
        z = z_inicial

        #Listas para armazenamento
        idx_suporte = [range(0, n_samples)]
        erros = []
        mult_lagrange = []

        #Loop de iteração
        for k in range(1, max_iterations + 1):

            #Calculando o erro associado
            erro = B - np.matmul(A, z)

            #Atualização
            z_anterior = z
            z = z + np.matmul(linalg.inv(np.matmul(A.T, A) + step_size * np.diag(np.matmul(A.T, A))), np.matmul(A.T, erro))

            #Condição para manter a atualização
            erro_novo = B - np.matmul(A, z)
            if np.mean(erro_novo**2) < np.mean(erro**2):
                z = z
            else:
                mu = mu/10
                z = z_anterior
            
            #Condição para a janela de poda
            if k > window_size and k < max_iterations - window_size:
                
                #Realização da poda
                n_colunas_removidas = int((n_samples - (n_samples * percentage))/(max_iterations - (2 * window_size)))

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                idx_remover = np.argsort(np.abs(np.squeeze(z)))[:n_colunas_removidas-1]

                #Armazenando os índices dos vetores de suporte a cada iteração
                idx_suporte.append(np.argsort(np.abs(np.squeeze(z)))[n_colunas_removidas:])

                #Remoção das colunas de A e linhas de z
                A = np.delete(A, idx_remover, axis = 1)
                z = np.delete(z, idx_remover, axis = 0)


            #Outra condição
            if k == max_iterations - window_size:

                #Realização da poda
                n_colunas_removidas = int(((n_samples - (n_samples * percentage))/(max_iterations - (2 * window_size))) + ((n_samples - n_samples * percentage)%(max_iterations - (2 * window_size))))

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                idx_remover = np.argsort(np.abs(np.squeeze(z)))[:n_colunas_removidas-1]

                #Armazenando os índices dos vetores de suporte a cada iteração
                idx_suporte.append(np.argsort(np.abs(np.squeeze(z)))[n_colunas_removidas:])

                #Remoção das colunas de A e linhas de z
                A = np.delete(A, idx_remover, axis = 1)
                z = np.delete(z, idx_remover, axis = 0)


            #Armazenando o erro
            erros.append(np.mean(erro**2))

            #Critério de parada
            if np.abs(np.mean(erro_novo**2)) < tolerance:
                break

        z_flat = z.ravel()
        mult_lagrange = np.zeros(n_samples)
        
        last_idx = np.array(list(idx_suporte[-1]))
        n_restantes = len(z_flat) - 1  # excluindo o bias
        mult_lagrange[last_idx[:n_restantes]] = z_flat[1:]

        self.X_treino = X
        self.alphas = mult_lagrange
        
        #Resultados
        resultados = {"mult_lagrange": mult_lagrange,
                    "b": np.squeeze(z)[0],
                    "Erros": erros,
                    "Indices_multiplicadores": idx_suporte}

        #Retornando os multiplicadores de Lagrange finais
        return(resultados)

    
    def predict(self,
                X_teste: NDArray[np.float64])-> NDArray[np.float64]:
    
            return np.sign(self.kernel.compute(X_teste, self.X_treino) @ self.alphas)