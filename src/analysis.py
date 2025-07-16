

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


import matplotlib.pyplot as plt
from scipy.stats import norm

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
    plt.show()




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
