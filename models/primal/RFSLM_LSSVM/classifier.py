'''
Implemetação da classe para o método RFSLM-LSSVM
Data: 14/11/2024
'''

#------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#------------------------------------------------------------------------------
import numpy as np
from numpy.typing import NDArray
from numpy import linalg
import pandas as pd
from sklearn.model_selection import train_test_split

from contracts.base import Model
from kernel.registry import build_kernel
from models.base import BuildKKTSystem

class ReducedFixedSizedLMClassifier(Model):
    '''Classe cconcreta para a implementação do modelo
    Reduced Fixed Sized Levenberg-Marquardt LSSVM'''
    def __init__(self,
                     tikhonov_regulraization: float = 0.2,
                     kernel: str = 'rbf',
                     **kernel_params)-> None:
            self.tikhonov_regularization = tikhonov_regulraization
            self.kernel = build_kernel(kernel=kernel, kernel_params=kernel_params)

    
    #----------------------------------------------------------------
    #Implementação da Proposta 
    #Reduced Set Fixed Sized Levenberg-Marquardt LSSVM (RFSLM-LSSVM)
    #Source:https://www.researchgate.net/publication/377073903_New_Iterative_Pruning_Methods_for_Least_Squares_Support_Vectors_Machines
    #----------------------------------------------------------------
    #Implementação do Método Fit() para a proposta RFSLM-LSSVM
    def fit(self,
            X: NDArray,
            y: NDArray,
            max_iterations: int = 100,
            window_size: int = 5,
            step_size: float = 0.1,
            percentage: float = 0.5,
            tolerance: float = 0.001,
            cut_reduction: float = 0.1):
        """
        Interface do método
        Ação: Este método visa realizar a poda iterativa dos vetores
        de suporte estimados no LSSVM, trabalhando com a matriz dos
        coeficientes reduzida em cada iteração.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        step_size: Taxa de aprendizado (escalar);
        cut_reduction: Porcentagem de redução por iteração de corte (escalar);
        percentage: Percentual de remoção (escalar);
        max_iterations: Número máximo de iterações (escalar);
        tolerance: Tolerância (Critério de parada);
        window_size: Termo que define a faixa de poda (escalar).

        OUTPUT:
        vetor esparso de multiplicadores de Lagrange estimados. 
        """

        #Gerando os indices para os vetores de suporte
        indices = np.arange(0, X.shape[0])

        #Convertendo array de entrada em Dataframe
        X = pd.DataFrame(X)
        y = pd.DataFrame(y)

        #Inicialização aleatória do conjunto com vetores de suporte
        X_rest, X_vs, y_rest, y_vs, indices_rest, indices_vs = train_test_split(X, y, indices, test_size= 1-percentage)
        X_vs = np.array(X_vs)
        y_vs = np.squeeze(np.array(y_vs))

        #Inicialização aleatória do vetor solução do método de Levenberg-Marquardt
        z_inicial = np.random.normal(loc=0, scale=1, size=(X_vs.shape[0] + 1, 1))
        z = z_inicial

        #Listas para armazenamento
        erros = []
        mult_lagrange = []
        idx_suporte = [range(0, X.shape[0])]

        #Loop de iteração
        for k in range(1, max_iterations+1):
            
            #Cálculo da matriz A e vetor b para o conjunto de vetores suporte
            support_vector_kernel = build_kernel(kernel='rbf', kernel_params={'gamma': 0.5})
            builder = BuildKKTSystem(kernel_matrix=support_vector_kernel, tikhonov_regularization=self.tikhonov_regularization)
            A, B = builder.build_classification(X_vs, y_vs)

            #Cálculo do erro para a k-ésima iteração
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
            
            if k in np.arange(window_size, max_iterations, window_size, dtype='int'):

                #Número de vetores suporte removidos dados pela porcentagem de redução por iteração de corte
                numero_vetores_suporte_removidos = int(cut_reduction * X_vs.shape[0])

                #Ordenar os menores valores absolutos dos multiplicadores de Lagrange
                idx_remover = np.argsort(np.abs(np.squeeze(z[:len(z)-1])))[:numero_vetores_suporte_removidos]

                #Remoção dos vetores suporte de X_vs, multiplicadores de z e índices de vetores de suporte
                vetores_suporte_removidos = X_vs[idx_remover]
                X_vs = np.delete(X_vs, idx_remover, axis = 0)
                y_vs = np.delete(y_vs, idx_remover)
                z = np.delete(z, idx_remover, axis = 0)

                #Remoção dos indices relacionados aos vetores de suporte
                indices_removidos = indices_vs[idx_remover]
                indices_vs = np.delete(indices_vs, idx_remover)

                #Seleção aleatória de Percent dados para serem
                #adicionados ao conjunto de vetores de suporte
                X_add, _, y_add, _, indices_add, _ = train_test_split(X_rest, y_rest, indices_rest, train_size=numero_vetores_suporte_removidos)
                X_vs = np.append(X_vs, X_add, axis = 0)
                y_vs = np.append(y_vs, y_add)
                indices_vs = np.append(indices_vs, indices_add)

                #Atualização do conjunto de vetores que não são de suporte
                #Remoção
                X_rest = X_rest.loc[~X_rest.index.isin(np.squeeze(indices_add))]
                X_rest.shape
                indices_rest = np.setdiff1d(indices_rest, indices_add)
                len(indices_rest)

                #Adição
                X_rest = pd.concat((X_rest, pd.DataFrame(vetores_suporte_removidos,
                                                        index = indices_removidos)), axis = 0)
                X_rest.shape
                indices_rest = np.append(indices_rest, indices_removidos)
                len(indices_rest)

                #Inicialização aleatória do vetor z_t P%
                z_percent = np.random.normal(loc=0, scale=1, size=(numero_vetores_suporte_removidos, 1))

                #Atualização dos multiplicadores de Lagrange
                z = np.append(z, z_percent, axis=0)
                
            
            #Armazenando o erro e os indices de vetores de suporte
            erros.append(np.mean(erro**2))
            idx_suporte.append(indices_vs)

            #Critério de parada
            if np.abs(np.mean(erro_novo**2)) < tolerance:
                break


        mult_lagrange = np.zeros(X.shape[0])
        mult_lagrange[indices_vs] = np.squeeze(z)[1:len(z)]

        self.X_treino = X
        self.alphas = mult_lagrange
        
        #Resultados
        resultados = {"mult_lagrange": mult_lagrange,
                    "b": np.squeeze(z)[0],
                    "Erros": erros,
                    "Indices_multiplicadores": idx_suporte}
        
        return(resultados)
    
    
    def predict(self,
                X_teste: NDArray[np.float64])-> NDArray[np.float64]:
    
            return np.sign(self.kernel.compute(X_teste, self.X_treino) @ self.alphas)