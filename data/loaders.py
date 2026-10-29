'''Script para implementação de classes concretas
para carregamento de dados'''

from pathlib import Path
import pandas as pd
from contracts.base import DataLoader, DataSourceConfig


class PandasDataLoader(DataLoader):
    '''Classe concreta para carregamento de arquivos.csv e .xlsx
    em pandas'''
    def load_data(self, config: DataSourceConfig):
        extension = Path(config.path).suffix
        if extension == '.csv':
            return pd.read_csv(config.path)
        return pd.read_excel(config.path)
