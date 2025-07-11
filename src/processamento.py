'''
← Carregamento e preparação dos dados

'''
# src/processamento.py

import sys
import os
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.model_selection import train_test_split
from src.plot import plotar_distribuicao_treino_teste
from sklearn.model_selection import train_test_split

def carregar_dados(caminho):
    """
    Carrega um arquivo pickle e retorna um DataFrame.
    """
    try:
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        
        dados = pd.read_pickle(caminho)

        if not isinstance(dados, pd.DataFrame):
            raise ValueError("O arquivo pickle não contém um DataFrame")

        print(f"✅ Dados carregados com sucesso! Shape: {dados.shape}")
        return dados

    except Exception as e:
        print(f"❌ Erro ao carregar {caminho}: {str(e)}")
        raise


def separar_dados_treino_teste(dados, target='dst', test_size=0.3, random_state=100, plot=False):
    """
    Separa os dados em conjuntos de treino e teste, com opção de visualizar a divisão.

    Parâmetros:
    -----------
    dados : DataFrame
        DataFrame com todas as variáveis
    target : str, opcional
        Nome da variável alvo (padrão 'dst')
    test_size : float, opcional
        Proporção de dados para o conjunto de teste (padrão 0.3)
    random_state : int, opcional
        Semente para reprodutibilidade (padrão 100)
    plot : bool, opcional
        Se True, gera gráfico da divisão treino/teste (padrão False)

    Retorna:
    --------
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = dados.drop(columns=[target])
    y = dados[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if plot:
        plotar_distribuicao_treino_teste(X_train, X_test)

    return X_train, X_test, y_train, y_test



def preparar_dados_transformer(dados, coluna_alvo='dst', test_size=0.3, random_state=42, salvar_plot=True):
    """
    Separa os dados em treino e teste. Pode salvar gráfico da divisão.
    """
    if coluna_alvo not in dados.columns:
        raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados.")

    X = dados.drop(columns=[coluna_alvo])
    y = dados[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if salvar_plot:
        plotar_distribuicao_treino_teste(X_train, X_test)

    return X_train, X_test, y_train, y_test




