'''
Treinamento, tuning

'''
# src/arvore.py

import logging
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

from src.avaliacao import avaliar_modelo
from src.processamento import separar_dados_treino_teste
from src.plot import (
    plotar_comparacao_modelos,
    exibir_model,
    plotar_distribuicao_treino_teste
)
from src.analysis import (
    estimar_modelo,
    diagnosticar_residuos,
    analisar_multicolinearidade,
    analise_splits
)

 
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
      
    
    # Preparar dados para análise
    dados_arvore = X_train.copy()
    dados_arvore['target'] = y_train
    df_splits = None

    if modelo_tipo:
        # Modo de execução para um único modelo específico
        if modelo_tipo == 'decision_tree':
            if usar_tuning:
                modelo, best_params = tunar_decision_tree(X_train, y_train, **kwargs)
            else:
                modelo = treinar_decision_tree(X_train, y_train, **kwargs)
                best_params = None

        elif modelo_tipo == 'random_forest':
            if usar_tuning:
                modelo, best_params = tunar_random_forest(X_train, y_train, **kwargs)
            else:
                modelo = treinar_random_forest(X_train, y_train, **kwargs)
                best_params = None

        elif modelo_tipo == 'gradient_boosting':
            if usar_tuning:
                modelo, best_params = tunar_gradient_boosting(X_train, y_train, **kwargs)
            else:
                modelo = treinar_gradient_boosting(X_train, y_train, **kwargs)
                best_params = None

        else:
            raise ValueError(f"Modelo '{modelo_tipo}' não reconhecido.")
            
        # Análise de splits apenas para árvores individuais
        if modelo_tipo in ['decision_tree', 'random_forest'] and hasattr(modelo, 'tree_'):
            from src.analysis import analise_splits
            df_splits = analise_splits(modelo, X_train, y_train)
            logging.info("\n📌 Análise de splits:")
            logging.info(df_splits.head())

        return modelo, best_params, df_splits
        
    else:
        # Modo de execução padrão para múltiplos modelos
        modelo_rf = treinar_random_forest(X_train, y_train)
        modelo_gb = treinar_gradient_boosting(X_train, y_train)

        if hasattr(modelo_rf, 'estimators_'):
            from src.analysis import analise_splits
            df_splits = analise_splits(modelo_rf.estimators_[0], X_train, y_train)
            logging.info("\n📌 Análise de splits:")
            logging.info(df_splits.head())

        return modelo_rf, modelo_gb, df_splits


# ====== FUNÇÕES DE TREINAMENTO ======

def treinar_decision_tree(X_train, y_train, max_depth=3, random_state=42, **kwargs):
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


# ====== FUNÇÕES DE TUNING ======

def tunar_decision_tree(X_train, y_train, random_state=42, **kwargs):
    """Realiza tuning de hiperparâmetros para Decision Tree."""
    param_grid = {
        'max_depth': [2, 4, 6, 8],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        **{k: [v] for k, v in kwargs.items()}  # Inclui parâmetros adicionais
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


def tunar_random_forest(X_train, y_train, random_state=42, **kwargs):
    """Realiza tuning de hiperparâmetros para Random Forest."""
    param_grid = {
        'n_estimators': [50, 100, 150],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5],
        **{k: [v] for k, v in kwargs.items()}
    }

    grid = GridSearchCV(
        estimator=RandomForestRegressor(random_state=random_state),
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='neg_mean_squared_error'
    )
    grid.fit(X_train, y_train)
    return grid.best_estimator_, grid.best_params_


def tunar_gradient_boosting(X_train, y_train, random_state=42, **kwargs):
    """Realiza tuning de hiperparâmetros para Gradient Boosting."""
    param_grid = {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5],
        'min_samples_split': [2, 5],
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