'''Script para implementação de funções
 utilitárias para o modelo TCSMO_LSSVM'''

#-------------------------------------
# Carregando pacotes relevantes
#-------------------------------------
import numpy as np
import numpy.typing as nptype


def FGWSS(gradient: nptype.NDArray[np.float64],
          K: nptype.NDArray[np.float64]):
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
        
    #Primeiro índice
    i = np.argmax(np.absolute(gradient))

    #Segundo índice
    exception = i
    m = np.zeros(gradient.shape[0], dtype=bool)
    m[exception] = True

    numerador = (gradient-gradient[i])**2
    denominador = 2 * (K[i, i] * np.ones(K.shape[0]) + np.diag(K) - K[i, :] - K[:, i])
    quociente = np.ma.array(numerador/denominador, mask=m)
    j = np.argmax(quociente)

    return (i, j)