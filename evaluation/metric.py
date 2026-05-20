'''Script para o cálculo de métricas gerais
para as tarefas de regressão e classificação'''

#--------------------------------------------------------
# Carregando pacotes relevantes
#--------------------------------------------------------
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    # Classificação
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,

    # Regressão
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from contracts.base import Evaluator, Data

#--------------------------------------------------------
# Implementação das classes
#--------------------------------------------------------
class ClassificationEvaluator(Evaluator):
    '''Classe para a avaliação de modelos
    em tarefas de classficação'''

    def evaluate(self, target: Data, estimated: Data)-> Dict[str, Any]:
        metrics: Dict[str, float] = {
            'accuracy': accuracy_score(y_true=target, y_pred=estimated),
            'precision': precision_score(y_true=target, y_pred=estimated),
            'recall': recall_score(y_true=target, y_pred=estimated),
            'F1 Score': f1_score(y_true=target, y_pred=estimated)
        },
        confusion = confusion_matrix(y_true=target, y_pred=estimated)
        report = classification_report(y_true=target, y_pred=estimated)

        return{'metrics': metrics,
                'confusion_matrix': confusion,
                'report': report}

class RegressionEvaluator(Evaluator):
    '''Classe para a avaliação de modelos
    em tarefas de regressão'''
    
    def evaluate(self, target: Data, estimated: Data)-> Dict[str, float]:
        return {'RMSE': np.sqrt(mean_squared_error(y_true=target, y_pred=estimated)),
                'MAE': mean_absolute_error(y_true=target, y_pred=estimated),
                'R²': r2_score(y_true=target, y_pred=estimated)}