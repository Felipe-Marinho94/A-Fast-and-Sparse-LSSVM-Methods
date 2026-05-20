'''
Implementação de classe para o ajuste (fit) e predict da
proposta TCSMO_LSSVM
'''

#-------------------------------------------------------------------------------------------
#Carregando alguns pacotes relevantes
#-------------------------------------------------------------------------------------------
from typing import Dict
import numpy as np
import numpy.typing as nptype
from contracts.base import Model
from kernel.registry import build_kernel
from models.dual.TCSMO_LSSVM.utils import FGWSS

#-------------------------------------------------------------------------------------------
#Implementando a classe
#-------------------------------------------------------------------------------------------
class ThreeTermConjugateSMOClassifier(Model):
    '''Classe para implementação do classificador Three Term
    Conjugate LSSVM'''
    def __init__(self, 
                 tikhonov_regularization: float = 2,
                 kernel: str ="rbf",
                 **kernel_params)-> None:
        self.tikhonov_regularization = tikhonov_regularization
        self.kernel = build_kernel(kernel=kernel, kernel_params=kernel_params)

    #--------------------------------------------------------------------------------
    #Implementação do método fit() para ajustar o LSSVM utilizando
    #uma variante do algoritmo SMO utilizando direções conjugadas de
    #três termos
    #SOURCE:https://www.sciencedirect.com/science/article/abs/pii/S0031320323001784
    #--------------------------------------------------------------------------------
    def fit(self,
            X: nptype.NDArray,
            y: nptype.NDArray,
            tolerance: float = 0.001,
            max_iterations: int = 100)-> Dict[str, nptype.NDArray]:
        '''
        Interface do método
        Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
        a metodologia CFGSMO, retornando os multiplicadores de lagrange
        ótimos para o problema de otimização.

        INPUT:
        X: Matriz de features (array N x p);
        y: Vetor de target (array N x 1);
        gamma: termo de regularização de Tichonov (escalar)
        kernel: função de kernel utilizada (string: "linear", "polinomial", "gaussiano");
        epsilon: Tolerância (Critério de parada)
        N: Número máximo de iterações

        OUTPUT:
        vetor ótimo de multiplicadores de Lagrange estimados.
        '''
        #Construindo a matriz de kernel
        n_samples, _ = X.shape 
        K = self.kernel.compute(X)
        
        #Regularização de Tichonov
        K_tilde = K + (1/self.tikhonov_regularization)*np.diag(np.full(K.shape[0], 1))
        
        #Inicialização
        #Multiplicadores de Lagrange
        alphas = np.zeros(n_samples)

        #Gradiente
        Gradient = -y

        #direções conjugadas
        z_penultimo = np.zeros(n_samples)
        z_ultimo = np.zeros(n_samples)

        #parâmetros nu
        nu_penultimo = np.zeros(n_samples)
        nu_ultimo = np.zeros(n_samples)

        #termo tau
        tau_penultimo = 1
        tau_ultimo = 1

        #Controle de iteração
        k = 0

        #erro
        erro = []

        while k <= max_iterations:
            
            #Seleção do conjunto de trabalho
            i, j = FGWSS(Gradient, K_tilde)

            #Cálculo do parâmetro conjugado
            delta_penultimo = (nu_penultimo[j] - nu_penultimo[i])/tau_penultimo
            delta_ultimo = (nu_ultimo[j] - nu_ultimo[i])/tau_ultimo

            #Construindo a direção descendente conjugada
            z = np.eye(1, n_samples, i)[0] - np.eye(1, n_samples, j)[0] + delta_penultimo * z_penultimo + delta_ultimo * z_ultimo

            #Atualização dos z
            z_ultimo = z_penultimo
            z_penultimo = z

            #Atualização dos vetores auxiliares
            nu = K_tilde[:, i] - K_tilde[:, j] + delta_penultimo * nu_penultimo + delta_ultimo * nu_ultimo
            nu_ultimo = nu_penultimo
            nu_penultimo = nu

            #Calculando o tau
            tau = nu[i] - nu[j]

            #Atualização dos taus
            tau_ultimo = tau_penultimo
            tau_penultimo = tau

            #Calculando o passo utilizando line search
            rho = (Gradient[j] - Gradient[i])/tau

            #Atualização dos multiplicadores de Lagrange
            alphas = alphas + rho * z

            #Atualização do gradiente
            Gradient = Gradient + rho * nu

            #Armazenando o erro
            erro.append(np.max(Gradient) - np.min(Gradient))

            #Condição de parada
            if np.abs(np.max(Gradient) - np.min(Gradient)) <= tolerance:
                break
            
        #Resultado
        resultados = {"lagrange": alphas,
                      "error": erro}
        
        return resultados
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para a primeira proposta considerando um
    #problema de classificação
    #------------------------------------------------------------------------------
    def predict(self,
                lagrange_multipliers: nptype.NDArray[np.float64],
                X_treino: nptype.NDArray[np.float64],
                X_teste: nptype.NDArray[np.float64])-> nptype.NDArray[np.float64]:
        #Inicialização
        estimado = np.zeros(X_teste.shape[0])
        
        #Construção da matriz de Kernel
        K= self.kernel.compute(X_teste, X_treino)
        
        #Cálculo da predição
        estimado =np.sign(K @ lagrange_multipliers)

        return estimado
