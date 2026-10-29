'''
Classe contendo algumas fuções úteis no desenvolvimento de outras classes
Data:15/11/2024
'''
import numpy as np
import numpy.typing as nptype


def FGWSS(grad: nptype.NDArray,
          K_tilde: nptype.NDArray) -> tuple[int, int]:
    '''
    Interface do Método
    Ação: Visa selecionar os índices dos dois multiplicadores de
    Lagrange que serão atualizados, por meio da heurística do
    método Functional Gain Working Selection Stategy (FGWSS)

    INPUT:
    gradient: Vetor Gradiente (array n)
    K: Matriz de Kernel (array n x n)

    OUTPUT:
    os dois índices selecionados
    '''
    n = len(grad)
    
    # Passo 1: Seleciona i como o índice do maior |g_l|
    i = np.argmax(np.abs(grad))
        
    # Passo 2: Seleciona j que maximiza o critério de direção conjugada
    max_score = -np.inf
    j = -1
        
    for l in range(n):
        if l == i:
            continue  # Não seleciona o mesmo índice
            
        # Numerador: (g_l - g_i)^2
        numerator = (grad[l] - grad[i]) ** 2
            
        # Denominador: 2 * (K̃_ii + K̃_ll - K̃_il - K̃_li)
        K_ii = K_tilde[i, i]
        K_ll = K_tilde[l, l]
        K_il = K_tilde[i, l]
        K_li = K_tilde[l, i]  # (simétrica, pode ser igual a K_il)
        denominator = 2 * (K_ii + K_ll - K_il - K_li)
            
        # Evita divisão por zero (adiciona um epsilon se necessário)
        if denominator < 1e-12:
            denominator = 1e-12
            
        score = numerator / denominator
            
        if score > max_score:
            max_score = score
            j = l
        
    return i, j
    
    
def gradient(alphas: nptype.NDArray[np.float64],
             K_tilde: nptype.NDArray[np.float64],
             y: nptype.NDArray[np.float64]) -> nptype.NDArray[np.float64]:
    '''
    Calcula o valor do gradiente da função objetivo
    INPUT:
    alphas - Multiplicadores de Lagrange (array n x 1)
    '''
    return(np.dot(K_tilde, alphas) - y)
    
    
def phi(d_anterior: nptype.NDArray,
         l_anterior: nptype.NDArray) -> float:
    '''
    Calcula o parâmetro phi utilzado nas iterações
    '''
    return(np.dot(d_anterior, l_anterior)/np.dot(l_anterior, l_anterior))


def rho_otimo(d_anterior: nptype.NDArray,
              gradiente_anterior: nptype.NDArray,
              l_anterior: nptype.NDArray,
              epsilon: float,
              p: float) -> float:
    '''
    Calcula o parâmetro rho ótimo utilzado nas iterações
    '''
    return(-np.dot(d_anterior, gradiente_anterior)/(epsilon * np.dot(l_anterior, l_anterior) * p))


def phi_barra(d_anterior: nptype.NDArray,
              l_anterior: nptype.NDArray) -> float:
    '''
    Calcula o parâmetro rho barra utilzado nas iterações
    '''
    return(np.dot(d_anterior, d_anterior)/(np.dot(d_anterior, l_anterior)))


def p(gradiente: nptype.NDArray,
      d_anterior: nptype.NDArray,
      l_anterior: nptype.NDArray) -> float:
    '''
    Calcula o parâmetro p utilzado nas iterações
    '''
    return(1- (np.dot(gradiente, d_anterior)**2)/(np.dot(gradiente, gradiente)*np.dot(d_anterior, d_anterior))+
            (np.dot(gradiente, l_anterior)/(np.linalg.norm(gradiente)*np.linalg.norm(l_anterior))+
                np.linalg.norm(gradiente)/np.linalg.norm(l_anterior))**2)


def beta_DY(gradiente: nptype.NDArray,
             d_anterior: nptype.NDArray,
             l_anterior: nptype.NDArray) -> float:
    '''
    Calcula o parâmetro beta DY utilzado nas iterações
    '''
    return(np.dot(gradiente, gradiente)/np.dot(d_anterior, l_anterior))


def measure(X_teste: nptype.NDArray[np.float64],
            y_treino: nptype.NDArray[np.float64],
            K: nptype.NDArray[np.float64],
            alphas: nptype.NDArray[np.float64]) -> nptype.NDArray[np.float64]:

        '''
        Método para calcular a medida de distância
        g(.) para um dado registro x

        INPUT:
        K - matriz de kernel (n x n);
        X - array representando o conjunto de treino (Nt xp);
        kernel - string indicando o kernel selecionado.

        OUTPUT:
        g(x) - medida de distância ao hiperplano (float).
        '''

        #Inicialização
        estimado = np.zeros(X_teste.shape[0])
        n_samples_teste = X_teste.shape[0]
    
        #Construção do vetor para os valores de g(x)
        for i in range(n_samples_teste):
                
            #Realização da predição
            estimado[i] = np.dot(alphas * y_treino, K[:, i])

        return estimado