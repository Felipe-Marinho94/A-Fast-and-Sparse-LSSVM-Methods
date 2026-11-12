'''
Implementação de classe para o ajuste (fit) e predict da
proposta SCG_LSSVM
'''

from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from contracts.base import Model
from kernel.registry import build_kernel
from models.dual.SCG_LSSVM.utils import (
    beta_DY, 
    gradient, 
    rho_otimo, 
    phi,
    phi_barra, 
    p,
    measure)


class SpectralGradientConjugateClassifier(Model):
    '''Classe para implementação do regressor Spectral Gradient
    Conjugate LSSVM com poda via ganho funcional'''
    def __init__(self, 
                 tikhonov_regularization: float = 0.2,
                 kernel: str ="rbf",
                 pruning_percentual: float = 0.2,
                 **kernel_params)-> None:
            self.tikhonov_regularization = tikhonov_regularization
            self.kernel = build_kernel(kernel=kernel, kernel_params=kernel_params)
            self.pruning_percentual = pruning_percentual

    #---------------------------------------------------------------
    #Função para ajustar o modelo LSSVM utilizando um novo método dos
    #gradientes conjugados espectrais
    #SOURCE: https://journalofinequalitiesandapplications.springeropen.com/articles/10.1186/s13660-020-02375-z
    #---------------------------------------------------------------
    def fit(self,
            X: NDArray,
            y: NDArray,
            tolerance: float = 0.001,
            max_iterations: int = 100,
            validation_split: float = 0.2)-> Dict[str, NDArray | list[float]]:
        '''
            Interface do método
            Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
            a metodologia SCFGSMO, retornando os multiplicadores de lagrange
            ótimos para o problema de otimização.

            INPUT:
            X: Matriz de features (array N x p);
            y: Vetor de target (array N x 1);
            Nr: Número de vetores de suporte a serem removidos a cada iteração (escalar)
            tolerance: Tolerância (Critério de parada)
            max_iterations: Número máximo de iterações
            percentage: Percentual de poda via ganho funcional

            OUTPUT:
            vetor ótimo de multiplicadores de Lagrange estimados.
        '''

        # Obtenção dos X_valid e y_valid
        X, X_valid, y, y_valid = train_test_split(X, y, test_size=validation_split, stratify=y)
        
        #Construindo a matriz de kernel
        n_samples, _ = X.shape 
        K = self.kernel.compute(X)
                
        #Regularização de Tikhonov
        K_tilde = K + (1/self.tikhonov_regularization)*np.diag(np.full(K.shape[0], 1))

        #Inicialização dos multiplicadores Lagrange
        alphas = np.zeros(n_samples)

        #Gradiente
        gradiente = gradient(alphas, K_tilde, y)

        #Direção conjugada inicial
        s = -gradiente

        #iteração
        k = 0

        #erro
        erro = []

        #Loop de iteração
        while k <= max_iterations:

            #Critério de parada
            if np.linalg.norm(gradiente) <= tolerance:
                    break

            #cálculo do parâmetro rho usando busca linear de wolfe
            rho = -np.dot(s, gradiente)/np.dot(np.dot(s, K_tilde), s)

            #Atualização dos padrões mais próximos do hiperplano
            alphas_anterior = alphas.copy()
            gradiente_anterior = gradiente.copy()

            #-----------------------------------------------------------
            #Etapa de determinação da distância do padrão ao hiperplano
            #-----------------------------------------------------------
            #Cálculo do g(x)
            g = measure(X, y, K_tilde, alphas)

            #Selecionando os padrões mais próximos
            active = np.argsort(np.abs(g))[: int(np.ceil(self.pruning_percentual * n_samples))]
            
            alphas[active] = alphas[active] + rho * s[active]
            gradiente = gradient(alphas, K_tilde, y)

            #Armazenando o erro
            erro.append(np.linalg.norm(gradiente))

            #-----------------------------------------------------------
            #Avaliação da condição de parada da poda iterativa
            #----------------------------------------------------------
            #Cálculo da saída estimada
            y_hat_anterior = self._predict_with(alphas_anterior, X, X_valid)
            y_hat = self._predict_with(alphas, X, X_valid)

            #Determinação da acurácia
            acc_anterior = accuracy_score(y_hat_anterior, y_valid)
            acc = accuracy_score(y_hat, y_valid)

            if (acc - acc_anterior) > 0:
                if (acc - acc_anterior) < 0.02:
                    break
                else:
                    pass
            else:
                pass

            #-----------------------------------------------------------
            #Etapa de atualização
            #-----------------------------------------------------------
            #calculando os parâmetros theta e beta
            ##Calculando o parâmetro phi
            d_anterior = alphas - alphas_anterior
            l_anterior = gradiente - gradiente_anterior

            parameter_phi = phi(d_anterior, l_anterior)
            parameter_phi_barra = phi_barra(d_anterior, l_anterior)
            parameter_p = p(gradiente, d_anterior, l_anterior)
            parameter_rho_otimo = rho_otimo(d_anterior, gradiente_anterior, l_anterior, tolerance, parameter_p)
            parameter_beta_DY = beta_DY(gradiente, d_anterior, l_anterior)

            theta = max(min(parameter_rho_otimo, parameter_phi_barra), parameter_phi)
            beta = theta * parameter_beta_DY

            #Atualização da direção
            s = (-theta * gradiente) + (beta * d_anterior)

            #Incremento
            k = k + 1

        self.X_train = X
        self.alphas = alphas

        #Resultados
        resultados = {"mult_lagrange": alphas,
                    "erro": erro,
                    "iteração": k,
                    "d": d_anterior,
                    "l": l_anterior,
                    "active": active,
                    "theta": theta,
                    "beta": beta,
                    "rho": rho,
                    "s": s,
                    "alphas": alphas_anterior}

        return resultados
    
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