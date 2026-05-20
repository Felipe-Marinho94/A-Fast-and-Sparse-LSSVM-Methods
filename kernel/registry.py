'''Script para registry dos tipos de kernel'''

#-------------------------------------------
# Carregando pacotes relevantes
#-------------------------------------------
from kernel.linear import LinearKernel
from kernel.polynomial import PolynomialKernel
from kernel.rbf import RBFKernel

KERNEL_REGISTRY = {
    'linear': LinearKernel,
    'polynomial': PolynomialKernel,
    'rbf': RBFKernel
}

def build_kernel(kernel: str, **kernel_params):
    '''Função de construção de kernels'''
    if kernel.lower() not in list(KERNEL_REGISTRY.keys()):
        ValueError(f'Kernel {kernel} não regitrado. Disponíveis: {list(KERNEL_REGISTRY)}')
    return KERNEL_REGISTRY[kernel.lower()](**kernel_params)