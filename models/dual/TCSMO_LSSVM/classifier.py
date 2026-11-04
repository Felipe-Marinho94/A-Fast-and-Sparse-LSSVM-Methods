'''
Implementação de classe para o ajuste (fit) e predict da
proposta TCSMO_LSSVM
'''

from typing import Dict
import numpy as np
import numpy.typing as nptype
from contracts.base import Model
from kernel.registry import build_kernel
from models.dual.TCSMO_LSSVM.utils import FGWSS


class ThreeTermConjugateSMOClassifier(Model):
    '''Classe para implementação do classificador Three Term
    Conjugate LSSVM com poda via ganho funcional'''
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
            max_iterations: int = 100,
            Nr: int = 10,
            percentage: float = 0.7)-> Dict[str, nptype.NDArray | list[float]]:
        '''
        Interface do método
        Ação: Este método visa realizar o ajuste do modelo LSSVM empregando
        a metodologia TCSMO, retornando os multiplicadores de lagrange
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
        #Construindo a matriz de kernel
        n_samples, _ = X.shape 
        K = self.kernel.compute(X)
        
        #Regularização de Tikhonov
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

            # Incremento
            k += 1

        # Realização da poda via ganho funcional
        active: list[int] = list(range(n_samples))
        inactive: list[int] = []
        
        #Cálculo do termo F
        F = -y + K_tilde @ alphas

        #Inicialização do erro na poda
        erro_pruning: list[float] = []

        while (len(active) >= np.ceil(percentage * n_samples)):
    
            idx_remover = np.argsort((1/2) * ((alphas[active] ** 2) * np.diag(K_tilde[np.ix_(active, active)]))- alphas[active] * F[active])[:Nr-1]

            # Converte para índices globais antes de tudo
            removed_global = np.array(active)[idx_remover]

            active = np.array(list(set(active).difference(set(removed_global))))
            inactive.extend(removed_global.tolist())

            # Atualiza F subtraindo APENAS os recém-removidos
            alphas_removed = np.zeros(n_samples)
            alphas_removed[removed_global] = alphas[removed_global]
            F = F - K_tilde @ alphas_removed

            alphas_active = np.zeros(n_samples)
            alphas_active[active] = alphas[active]

            Gradient = K_tilde @ alphas_active - y
            erro_pruning.append(np.max(Gradient) - np.min(Gradient))


        self.X_treino = X
        self.alphas = alphas_active
        
        #Resultado
        resultados = {"mult_lagrange": alphas_active,
                    "erro_pruning": erro_pruning,
                    "erro": erro,
                    "idx_suporte": active,
                    'idx_n_suporte': inactive}
            
        return resultados
    
    #------------------------------------------------------------------------------
    #Implementação do método predict() para a primeira proposta considerando um
    #problema de classificação
    #------------------------------------------------------------------------------
    def predict(self,
                X_teste: nptype.NDArray[np.float64])-> nptype.NDArray[np.float64]:

        return np.sign(self.kernel.compute(X_teste, self.X_treino) @ self.alphas)
