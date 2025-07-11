'''
Preparar dados, treinar, registrar

'''
# src/model.py

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import logging

from src.redes_neurais import ( 
    TransformerRegressor, 
    prever_transformer
)

def preparar_dados(df, coluna_alvo):
    X = df.drop(columns=[coluna_alvo])
    y = df[coluna_alvo]
    return train_test_split(X, y, test_size=0.2, random_state=42)

def treinar_regressao(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def treinar_e_registrar(nome, func_treino, X_train, y_train, X_test, y_test, **kwargs):
    from .log import registrar_modelo  # Ajuste conforme seu projeto
    modelo = func_treino(X_train, y_train, **kwargs)
    metrics = registrar_modelo(nome, modelo, X_train, y_train, X_test, y_test)
    return modelo, metrics


import tensorflow as tf
from tensorflow.keras import optimizers, losses, callbacks

def treinar_transformer(X_train, y_train, epochs=20, batch_size=32, learning_rate=1e-4, validation_split=0.1, **kwargs):
    # Criar o modelo (ajuste os parâmetros conforme necessidade)
    input_dim = X_train.shape[1]
    model = TransformerRegressor(
        input_dim=input_dim,
        d_model=64,
        num_heads=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1
    )

    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss=losses.MeanSquaredError(),
        metrics=["mae"]
    )

    early_stop = callbacks.EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        mode='min'
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        verbose=1,
        callbacks=[early_stop]
    )

    return model
