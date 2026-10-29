'''Script para a realização de produtos internos
em kernel gaussianos'''

from typing import Dict, Optional

import numpy as np
import numpy.typing as nptype
from kernel.base import Kernel

class RBFKernel(Kernel):
    def __init__(self, kernel_params: Optional[Dict[str, float]] = None)-> None:
        super().__init__()
        kernel_params = kernel_params or {}
        self.sigma = kernel_params.get('sigma', 0.5)

    def compute(self,
                X: nptype.NDArray, 
                Y: nptype.NDArray = None)-> nptype.NDArray:
        if Y is None:
            Y = X
        
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y**2, axis=1).reshape(1, -1)

        dist = X_norm + Y_norm - 2 * X @ Y.T
        return np.exp(-self.sigma * dist) 
