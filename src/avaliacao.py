'''
Avaliação

'''
# src/avaliacao.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import OrderedDict


def prever_transformer(modelo, X):
    """
    Função para gerar previsões para modelos transformer.
    Aqui deve estar a lógica para fazer previsões com seu transformer.
    Substitua pelo código real de inferência do transformer.
    """
    # Exemplo dummy - substituir pela inferência real
    return modelo.predict(X)


def avaliar_modelo(modelo, X, y, tipo_modelo='sklearn'):
    """
    Avalia um modelo de regressão e retorna métricas de desempenho ordenadas por valor decrescente.

    Args:
        modelo: modelo treinado (deve ter o método .predict)
        X: dados preditores
        y: valores reais
        tipo_modelo: tipo do modelo ('sklearn', 'transformer')

    Returns:
        OrderedDict com R2, MSE, RMSE e MAE (maior valor primeiro)
    """
    if tipo_modelo not in ['sklearn', 'transformer']:
        raise ValueError(f"Tipo de modelo desconhecido: {tipo_modelo}. Use 'sklearn' ou 'transformer'.")

    try:
        if tipo_modelo == 'transformer':
            y_pred = prever_transformer(modelo, X)
        else:
            y_pred = modelo.predict(X)
    except Exception as e:
        print(f"❌ Erro ao prever: {e}")
        return OrderedDict()  # retorna dict vazio

    metricas = {
        'R2': r2_score(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred)
    }

    # R2 primeiro, os demais em ordem decrescente
    metricas_ordenadas = OrderedDict(
        sorted(metricas.items(), key=lambda x: (x[0] != 'R2', -x[1] if x[0] != 'R2' else -x[1]))
    )

    return metricas_ordenadas


def avaliar_transformer(modelo, X_test, y_test):
    """
    Avalia o modelo transformer e imprime as métricas principais.
    """
    y_pred = prever_transformer(modelo, X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}
