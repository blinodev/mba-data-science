

'''
← Análise de resíduos, VIF, etc.

'''
# src/analysis.py

import logging
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

# Estima modelo OLS com fórmula
def estimar_modelo(df, formula):
    return smf.ols(formula=formula, data=df).fit()

# Diagnóstico dos resíduos
def diagnosticar_residuos(modelo):
    residuos = modelo.resid
    logging.info(f"Resumo dos resíduos:\n{residuos.describe()}")

# VIF - multicolinearidade
def analisar_multicolinearidade(X):
    vif_data = pd.DataFrame()
    vif_data['variavel'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    return vif_data

# Análise de splits de árvore
def analise_splits(modelo, X_train=None, y_train=None):
    if isinstance(modelo, (DecisionTreeRegressor, DecisionTreeClassifier)):
        path = modelo.cost_complexity_pruning_path(X_train, y_train)
        df_splits = pd.DataFrame({
            'ccp_alpha': path.ccp_alphas,
            'impurity': path.impurities,
            'n_nodes': [modelo.tree_.node_count] * len(path.ccp_alphas),
            'depth': [modelo.tree_.max_depth] * len(path.ccp_alphas)
        })

    elif isinstance(modelo, (RandomForestRegressor, RandomForestClassifier)):
        dados_arvores = [{
            'tree_id': i,
            'n_nodes': arvore.tree_.node_count,
            'depth': arvore.tree_.max_depth,
            'n_leaves': arvore.tree_.n_leaves
        } for i, arvore in enumerate(modelo.estimators_)]
        df_splits = pd.DataFrame(dados_arvores)

    elif isinstance(modelo, (GradientBoostingRegressor, GradientBoostingClassifier)):
        arvores = modelo.estimators_.ravel() if hasattr(modelo.estimators_, 'ravel') else modelo.estimators_
        dados_arvores = [{
            'stage': i,
            'n_nodes': arvore.tree_.node_count,
            'depth': arvore.tree_.max_depth,
            'n_leaves': arvore.tree_.n_leaves
        } for i, arvore in enumerate(arvores)]
        df_splits = pd.DataFrame(dados_arvores)
        if hasattr(modelo, 'train_score_'):
            df_splits['train_score'] = modelo.train_score_

    else:
        raise ValueError("Modelo não suportado para análise de splits.")

    
    print(df_splits.head())
    return df_splits
