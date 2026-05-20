'''Script para adição de funções de pré-processamento
que podem ser adicionadas na pipeline'''

#----------------------------------------
# Carregando pacotes relevantes
#----------------------------------------
from typing import List, Any, TypeVar
import numpy as np
import numpy.typing as nptype
import pandas as pd


#----------------------------------------
# Implementação de funções de
# pré-processamento
#----------------------------------------
# Análise Univariada
def drop_nan(df: pd.DataFrame)-> pd.DataFrame:
    '''Remove registros (linhas) com valores nulos
    INPUT:
    dataset: Dataframe com os dados de entrada (Dataframe);

    OUTPUT:
    Dataframe sem as linhas com valores nulos.'''
    return df.dropna(axis=0)


def volumetry_filter(df: pd.DataFrame, threshold: float)-> pd.DataFrame:
    '''função para remover features com um percentual de nulos
    acima do valor do threshold
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volumetria
    '''
    for column in df.columns:
        if df[column].isna().sum()/df.shape[0] > threshold:
            df = df.drop([column], axis = 1)
    
    return df

def volatility_filter(df: pd.DataFrame, threshold: float)-> pd.DataFrame:
    '''função para remover features com baixa variabilidade, que por
    sua vez, não contribuem significativamente para a discriminação
    interclasses
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features com boa volatilidade'''
    
    for column in df.select_dtypes(include=["int64", "float64"]).columns.values:
        if round(df[column].var(), 2) < threshold:
            df = df.drop([column], axis = 1)
    
    return df

# Análise Bivariada
def filter_correlation(df: pd.DataFrame, threshold: float)-> pd.DataFrame:
    '''
    função para remover features colineares com base na correlação de spearmen
    INPUT:
        df - Pandas dataframe com as features a serem analisadas;
        threshold - Limiar utilizado como critério para a remoção
        da feature
    
    OUTPUT:
        pandas dataframe somente com as features descorrelacionadas
    '''

    corr_matrix = df.corr(method="spearman")
    iters = range(len(corr_matrix.columns))
    drop_cols: List[Any] = []

    for i in iters:
        for j in range(i+1):
            item = corr_matrix.iloc[j:(j+1), (i+1):(i+2)]
            col = item.columns
            row = item.index
            val = abs(item.values)

            if val >= threshold:
                drop_cols.append(col.values[0])
    
    drops = set(drop_cols)
    df = df.drop(columns=drops)
    print('Drop Column {}'.format(drops))
    return df


T = TypeVar('T', bound=np.generic)
def ensure_array(series: pd.Series[T])-> nptype.NDArray:
    '''Função para conversão de pd.Series para numpy array'''
    return series.to_numpy()


