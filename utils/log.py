# -*- coding: utf-8 -*-
"""
Created on Tue Jul 15 11:09:37 2025

Logging e relatórios

"""


# log.py

import logging
import os
from datetime import datetime
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error


class CleanConsoleFormatter(logging.Formatter):
    """Formatter personalizado para saída no console (sem timestamps/níveis)"""
    def format(self, record):
        return record.getMessage()


def configurar_logging():
    """Configura o sistema de logging com formatação diferenciada para console e arquivo."""
    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/model_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    # Configurar o logger principal
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Remover handlers existentes se houver
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Handler para arquivo (formato completo)
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    
    # Handler para console (formato limpo)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CleanConsoleFormatter())
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return log_file

import logging
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

def registrar_modelo(nome_modelo, modelo, X_train, y_train, X_test, y_test):
    """
    Avalia e registra o desempenho de um modelo de machine learning.
    
    Args:
        nome_modelo (str): Nome descritivo do modelo
        modelo: Modelo treinado com método predict()
        X_train: Dados de treino (features)
        y_train: Target de treino
        X_test: Dados de teste (features)
        y_test: Target de teste
        
    Returns:
        dict: Dicionário com métricas de treino e teste
        
    Raises:
        ValueError: Se os tamanhos dos dados não forem consistentes
    """
    # Validação dos dados
    if len(X_train) != len(y_train):
        raise ValueError("X_train e y_train têm tamanhos diferentes")
    if len(X_test) != len(y_test):
        raise ValueError("X_test e y_test têm tamanhos diferentes")

    # Predições
    y_pred_train = modelo.predict(X_train)
    y_pred_test = modelo.predict(X_test)

    # Cálculo das métricas
    metrics_train = {
        'R²': round(r2_score(y_train, y_pred_train), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_train, y_pred_train)), 4)
    }

    metrics_test = {
        'R²': round(r2_score(y_test, y_pred_test), 4),
        'RMSE': round(np.sqrt(mean_squared_error(y_test, y_pred_test)), 4)
    }

    # Formatação visual melhorada
    border = "═" * 50
    log_messages = [
        f"\n{border}",
        f"📊 PERFORMANCE: {nome_modelo.upper():^38}",
        border,
        f"🔵 TREINO │ R²: {metrics_train['R²']:>7} │ RMSE: {metrics_train['RMSE']:>10}",
        f"🔴 TESTE  │ R²: {metrics_test['R²']:>7} │ RMSE: {metrics_test['RMSE']:>10}",
        f"{border}\n"
    ]
    
    for message in log_messages:
        logging.info(message)

    return {'train': metrics_train, 'test': metrics_test}

