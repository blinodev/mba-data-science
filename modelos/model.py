
"""
Created on Tue Jul 15 12:09:57 2025

@author: familia
"""
# modelos.py
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0 = todos os logs, 1 = info removido, 2 = warning removido, 3 = error removido
import tensorflow as tf
import logging
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import LinearRegression
from utils.log import configurar_logging, registrar_modelo 

# Função para treinar regressão linear
def treinar_regressao(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

# Função para treinar e registrar métricas
def treinar_e_registrar(nome, func_treino, X_train, y_train, X_test, y_test, **kwargs):
    modelo = func_treino(X_train, y_train, **kwargs)
    metrics = registrar_modelo(nome, modelo, X_train, y_train, X_test, y_test)
    return modelo, metrics

def treinar_e_registrar_transformer(nome, func_treino, X_train, y_train, X_test, y_test, **kwargs):
    modelo, history = func_treino(X_train, y_train, **kwargs)
    metrics = registrar_modelo(nome, modelo, X_train, y_train, X_test, y_test)
    return modelo, history, metrics

# Estima modelo OLS com fórmula
def estimar_modelo(df, formula):
    return smf.ols(formula=formula, data=df).fit()


import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingRegressor


def diagnosticar_residuos(modelo):
    """
    Exibe diagnóstico dos resíduos:
    - Estatísticas descritivas
    - Histograma com curva de distribuição normal ajustada
    """
    residuos = modelo.resid
    mu = residuos.mean()
    std = residuos.std()

    logging.info(f"\n📊 Resumo dos resíduos:\n{residuos.describe()}")

    # Plotando histograma com curva normal
    plt.figure(figsize=(8, 6))
    plt.hist(residuos, bins=30, density=True, alpha=0.8, color='blue')

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 1000)
    p = norm.pdf(x, mu, std)
    plt.plot(x, p, linewidth=3, color='red')

    plt.xlabel('Resíduos', fontsize=11)
    plt.ylabel('Densidade', fontsize=11)

    # Removendo bordas superior e direita
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    plt.tight_layout()
    


def verificar_tamanhos(X_train, y_train, X_test=None, y_test=None):
    """
    Verifica se X e y têm o mesmo número de linhas nos conjuntos de treino e teste.

    Args:
        X_train: Features do treino (DataFrame ou array)
        y_train: Target do treino (Series ou array)
        X_test: Features do teste (opcional)
        y_test: Target do teste (opcional)

    Raises:
        ValueError: se algum par (X, y) não tiver o mesmo número de linhas.
    """
    if X_train.shape[0] != y_train.shape[0]:
        raise ValueError(f"Tamanho inconsistente no treino: X_train tem {X_train.shape[0]} linhas, y_train tem {y_train.shape[0]} linhas.")

    if X_test is not None and y_test is not None:
        if X_test.shape[0] != y_test.shape[0]:
            raise ValueError(f"Tamanho inconsistente no teste: X_test tem {X_test.shape[0]} linhas, y_test tem {y_test.shape[0]} linhas.")


from sklearn.tree import DecisionTreeRegressor
import pandas as pd

def analise_splits(tree_reg, X_train, y_train, random_state=42):
    """
    Analisa os resultados do pruning path (poda) do DecisionTreeRegressor.

    Args:
        tree_reg: modelo DecisionTreeRegressor já treinado (pode ter max_depth definido)
        X_train: dados de treino (features)
        y_train: target de treino
        random_state: semente para reproducibilidade nos treinamentos

    Returns:
        pd.DataFrame com colunas ['ccp_alpha', 'impureza_total', 'n_leaves'] ordenado do mais complexo ao mais simples
    """
    path = tree_reg.cost_complexity_pruning_path(X_train, y_train)

    tree_split = pd.DataFrame({
        "ccp_alpha": path.ccp_alphas,
        "impureza_total": path.impurities
    })

    tree_split["n_leaves"] = [DecisionTreeRegressor(random_state=random_state, ccp_alpha=alpha)
                              .fit(X_train, y_train).get_n_leaves()
                              for alpha in path.ccp_alphas]

    # Ordenar do modelo mais complexo (mais folhas) para o mais simples (menos folhas)
    tree_split.sort_values(by='n_leaves', ascending=False, inplace=True)
    tree_split.reset_index(drop=True, inplace=True)
    
    return tree_split

from sklearn.ensemble import RandomForestRegressor
import logging

def executar_modelos_arvores(X_train, y_train, modelo_tipo=None, usar_tuning=False, **kwargs):
    """
    Executa e avalia modelos baseados em árvores com opção de tuning.

    Args:
        X_train: Dados de treino (features)
        y_train: Target de treino
        modelo_tipo: Tipo de modelo ('decision_tree', 'random_forest', 'gradient_boosting')
        usar_tuning: Se True, realiza hyperparameter tuning
        **kwargs: Parâmetros adicionais para os modelos

    Returns:
        tuple: (modelo, best_params, df_splits) ou (modelo_rf, modelo_gb, df_splits)

    Raises:
        ValueError: Se modelo_tipo não for reconhecido
    """

    dados_arvore = X_train.copy()
    dados_arvore['target'] = y_train
    df_splits = None

    
    if modelo_tipo:
        if modelo_tipo == 'decision_tree':
            if usar_tuning:
                modelo, best_params = tunar_decision_tree(X_train, y_train, **kwargs)
                logging.info("✅ Decision Tree tuning com sucesso.")
            else:
                modelo = treinar_decision_tree(X_train, y_train, **kwargs)
                best_params = None
                logging.info("✅ Decision Tree treinada sem tuning.")

        elif modelo_tipo == 'random_forest':
            if usar_tuning:
                modelo, best_params = tunar_random_forest(X_train, y_train, **kwargs)
                logging.info("✅ Random Forest tuning com sucesso.")
            else:
                modelo = treinar_random_forest(X_train, y_train, **kwargs)
                best_params = None
                logging.info("✅ Random Forest treinada sem tuning.")

        elif modelo_tipo == 'gradient_boosting':
            if usar_tuning:
                modelo, best_params = tunar_gradient_boosting(X_train, y_train, **kwargs)
                logging.info("✅ Gradient Boosting tuning com sucesso.")
            else:
                modelo = treinar_gradient_boosting(X_train, y_train, **kwargs)
                best_params = None
                logging.info("✅ Gradient Boosting treinado sem tuning.")

        else:
            raise ValueError(f"❌ Modelo '{modelo_tipo}' não reconhecido.")

        if modelo_tipo == 'decision_tree' and hasattr(modelo, 'tree_'):
            df_splits = analise_splits(modelo, X_train, y_train)
            logging.info("📌 Análise de splits:")
            logging.info(df_splits.head())

        return modelo, best_params, df_splits

    else:
        logging.info("🧠 Executando Random Forest e Gradient Boosting sem tuning")

        modelo_rf = treinar_random_forest(X_train, y_train)
        modelo_gb = treinar_gradient_boosting(X_train, y_train)

        if isinstance(modelo_rf, RandomForestRegressor) and hasattr(modelo_rf, 'estimators_'):
            primeira_arvore = modelo_rf.estimators_[0]
            df_splits = analise_splits(primeira_arvore, X_train, y_train)
            logging.info("📌 Análise de splits da primeira árvore do Random Forest:")
            logging.info(df_splits.head())

        logging.info("✅ Random Forest e Gradient Boosting treinados sem tuning.")
        return modelo_rf, modelo_gb, df_splits

#############################################################################
#                  ====== FUNÇÕES DE TREINAMENTO ======                     #
#                                                                           #
#############################################################################


def treinar_decision_tree(X_train, y_train, max_depth=5, random_state=42, **kwargs):
    """Treina uma Decision Tree para regressão."""
    modelo = DecisionTreeRegressor(
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    modelo.fit(X_train, y_train)
    return modelo


def treinar_random_forest(X_train, y_train, n_estimators=100, max_depth=None, random_state=42, **kwargs):
    """Treina uma Random Forest para regressão."""
    modelo = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    modelo.fit(X_train, y_train)
    return modelo


def treinar_gradient_boosting(X_train, y_train, n_estimators=100, learning_rate=0.1, 
                             max_depth=3, random_state=42, **kwargs):
    """Treina um Gradient Boosting para regressão."""
    modelo = GradientBoostingRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        random_state=random_state,
        **kwargs
    )
    modelo.fit(X_train, y_train)
    return modelo

#############################################################################
#                  ====== FUNÇÕES DE TUNING ======                          #
#                                                                           #
#############################################################################


from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor

def tunar_decision_tree(X_train, y_train, random_state=100, verbose=2, **kwargs):
    """
    Realiza tuning de hiperparâmetros para Decision Tree usando GridSearchCV.

    Parâmetros:
        X_train (DataFrame): Conjunto de treino (features)
        y_train (Series): Target de treino
        random_state (int): Semente aleatória para reprodutibilidade
        verbose (int): Nível de verbosidade do GridSearchCV
        **kwargs: Parâmetros adicionais para o modelo

    Retorna:
        modelo_tunado: Instância treinada com os melhores hiperparâmetros
        melhores_parametros: Dicionário com os melhores hiperparâmetros
    """
    param_grid = {
        'max_depth': [3, 5, 10],
        'min_samples_split': [5, 10, 15],
        'min_samples_leaf': [5, 10],
        **{k: [v] for k, v in kwargs.items()}  # permite customizações
    }

    grid = GridSearchCV(
        estimator=DecisionTreeRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )

    grid.fit(X_train, y_train)

    return grid.best_estimator_, grid.best_params_


from sklearn.ensemble import RandomForestRegressor

def tunar_random_forest(X_train, y_train, random_state=42, **kwargs):
    """Realiza tuning de hiperparâmetros para Random Forest."""
    param_grid = {
        'n_estimators': [100, 500],
        'max_depth': [5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [30, 50]
        **{k: [v] for k, v in kwargs.items()}
    }

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=verbose,
        scoring='neg_mean_squared_error'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tunar_gradient_boosting(X_train, y_train, random_state=100, **kwargs):
    """Realiza tuning de hiperparâmetros para Gradient Boosting."""
    param_grid = {
        'n_estimators': [100, 500, 700],
        'learning_rate': [0.001, 0.01, 0.1],
        'max_depth': [3, 5],
        **{k: [v] for k, v in kwargs.items()}
    }

    grid = GridSearchCV(
        estimator=GradientBoostingRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


#############################################################################
#             Redes Neurais RNNs - Temporal Fusion transformer              #
#                        CARREGAMENTO DA BASE DE DADOS                      #
#############################################################################
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import EarlyStopping

def treinar_transformer(X_train, y_train, epochs=20, batch_size=32, learning_rate=1e-4, validation_split=0.1, **kwargs):
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

    early_stop = EarlyStopping(
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

    return model, history


from tensorflow.keras import layers, models, optimizers, losses

class TransformerRegressor(tf.keras.Model):
    def __init__(self, input_dim, d_model=64, num_heads=4, num_layers=2, dim_feedforward=128, dropout=0.1):
        super().__init__()
        self.input_proj = layers.Dense(d_model)
        self.encoder_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
            for _ in range(num_layers)
        ]
        self.ffn_layers = [
            models.Sequential([
                layers.Dense(dim_feedforward, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(d_model),
                layers.Dropout(dropout),
            ]) for _ in range(num_layers)
        ]
        self.layernorms1 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.layernorms2 = [layers.LayerNormalization(epsilon=1e-6) for _ in range(num_layers)]
        self.dropout = layers.Dropout(dropout)
        self.final_dense = layers.Dense(1)

    def call(self, x, training=False):
        # x shape: (batch_size, features)
        x = self.input_proj(x)  # (batch_size, d_model)
        x = tf.expand_dims(x, axis=1)  # (batch_size, seq_len=1, d_model)

        for mha, ffn, ln1, ln2 in zip(self.encoder_layers, self.ffn_layers, self.layernorms1, self.layernorms2):
            attn_output = mha(x, x, x, training=training)
            out1 = ln1(x + attn_output)
            ffn_output = ffn(out1, training=training)
            x = ln2(out1 + ffn_output)

        x = tf.squeeze(x, axis=1)  # (batch_size, d_model)
        out = self.final_dense(x)  # (batch_size, 1)
        return tf.squeeze(out, axis=1)  # (batch_size,)



import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def prever_transformer(model, X):
    preds = model.predict(X)
    return preds.squeeze()




