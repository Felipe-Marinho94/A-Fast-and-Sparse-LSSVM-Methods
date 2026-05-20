'''Script para desenvolvimento de abstrações/contratos
dos elementos do projeto'''

#--------------------------------------------------------
# Carregando pacotes relevantes
#--------------------------------------------------------
from abc import ABC, abstractmethod
from typing import TypeVar, Any, Dict, Optional
from dataclasses import dataclass


#--------------------------------------------------------
# Implementação das classes abstratas
#--------------------------------------------------------
Data = TypeVar('Data')
Output = TypeVar('Output')

class Model(ABC):
    '''Abstração/Contrato para modelos'''
    @abstractmethod
    def fit(self, X: Data, y: Data)-> Output:...

    @abstractmethod
    def predict(self, X: Data, y: Data)-> Output:...


class Evaluator(ABC):
    '''Abstração/Contrato para Avaliador'''
    @abstractmethod
    def evaluate(self, target: Data, estimated: Data)-> Output:...

@dataclass
class DataSourceConfig:
    '''Classe para configuração carregamento de dados'''
    path: str
    storage_options: Optional[Dict[str, Any]] = None 


class DataLoader(ABC):
    '''Abstração/Contrato para carregamento de dados'''
    @abstractmethod
    def load_data(self, config: DataSourceConfig)-> Data:...