.# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 15:17:00 2025

@author: Lino
"""

#%% Importar os pacotes

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
#from sklearn.ensemble import RandomForestRegressor
#from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#%% Importar a base (já tratada)
df = pd.read_pickle('df_final.pkl')

#%% 1. COMPREENSÃO DO PROBLEMA 
dados.info()
dados.head()

#%% # 2. COMPREENSÃO DOS DADOS - Verificar valores faltantes
def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

relatorio_missing(dados)



#%% # 3. PREPARAÇÃO DOS DADOS - Separando as variáveis Y e X

X = dados.drop(columns=['dst'])
y = dados['dst']


#%% Separando as amostras de treino e teste

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)


#%%######################### Árvore de Regressão ################################
#                                                                               #
###############################################################################


#%% # 4 - MODELAGEM - Gerando a árvore de decisão

tree_reg = DecisionTreeRegressor(max_depth=2, random_state=42)
tree_reg.fit(X_train, y_train)

#%% Gráfico de Barra para visualizar a distribuição com percentual e cores suaves
import numpy as np
import matplotlib.pyplot as plt

# Definindo os índices e nomes para o gráfico
ind = [0, 1]  # Dois períodos (Treinamento e Teste)
nomes = ["Treinamento", "Teste"]
largura = 0.75

# Contagem das amostras por período
contagens = {
    "train": len(X_train),
    "test": len(y_test)
}

# Cálculo do total e percentual
total = contagens["train"] + contagens["test"]
percent_train = (contagens["train"] / total) * 100
percent_test = (contagens["test"] / total) * 100

# Usando colormap "viridis" para cores suaves
cmap = plt.get_cmap("viridis")
cores = cmap(np.linspace(0.2, 0.8, 2))  # Pegando dois tons suaves

# Criando as barras
bar_train = plt.barh(ind[0], contagens["train"], largura, label=f"Treinamento ({percent_train:.1f}%)", color=cores[0])
bar_test = plt.barh(ind[1], contagens["test"], largura, label=f"Teste ({percent_test:.1f}%)", color=cores[1])

# Adicionando rótulos dentro das barras
plt.text(contagens["train"] / 2, ind[0], f"{percent_train:.1f}%", va='center', ha='center', color="white", fontsize=12, fontweight='bold')
plt.text(contagens["test"] / 2, ind[1], f"{percent_test:.1f}%", va='center', ha='center', color="black", fontsize=12, fontweight='bold')

# Ajustando o gráfico
plt.yticks(ind, nomes)
plt.ylabel("Período")
plt.xlabel("Quantidade de Amostras")
plt.title("Divisão de Treinamento/Teste", fontsize=14)
plt.legend()

# Exibindo o gráfico
plt.show()


#%% Plotando a árvore
plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_reg,
          feature_names=X.columns.tolist(),
          filled=True,
          node_ids=True,
          precision=2)
plt.show()


#%% Analisando os resultados dos splits

tree_split = pd.DataFrame(tree_reg.cost_complexity_pruning_path(X_train, y_train))
tree_split.sort_index(ascending=False, inplace=True)

print(tree_split)

#%% Obtendo os valores preditos pelo modelo

# Base de treinamento
tree_pred_train = tree_reg.predict(X_train)

# Base de teste
tree_pred_test = tree_reg.predict(X_test)

#%% Avaliando o modelo (base de treino)

mse_train_tree = mean_squared_error(y_train, tree_pred_train)
mae_train_tree = mean_absolute_error(y_train, tree_pred_train)
r2_train_tree = r2_score(y_train, tree_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_tree:.1f}")
print(f"RMSE: {np.sqrt(mse_train_tree):.1f}")
print(f"MAE: {mae_train_tree:.1f}")
print(f"R²: {r2_train_tree:.1%}")

#%% Avaliando o modelo (base de testes)

mse_test_tree = mean_squared_error(y_test, tree_pred_test)
mae_test_tree = mean_absolute_error(y_test, tree_pred_test)
r2_test_tree = r2_score(y_test, tree_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_tree:.1f}")
print(f"RMSE: {np.sqrt(mse_test_tree):.1f}")
print(f"MAE: {mae_test_tree:.1f}")
print(f"R²: {r2_test_tree:.1%}")

#%% Alguns hiperparâmetros do modelo

# max_depth: profundidade máxima da árvore
# min_samples_split: qtde mínima de observações para dividir o nó
# min_samples_leaf: qtde mínima de observações para ser nó folha

# Vamos aplicar um Grid Search
param_grid_tree = {
    'max_depth': [3, 5, 10],
    'min_samples_split': [5, 10, 15],
    'min_samples_leaf': [5, 10]
}

# Identificar o algoritmo em uso
tree_grid = DecisionTreeRegressor(random_state=42)

# Treinar os modelos para o grid search
tree_grid_model = GridSearchCV(estimator = tree_grid,
                               param_grid = param_grid_tree,
                               scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                               cv=5, verbose=2)

tree_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
tree_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
tree_best = tree_grid_model.best_estimator_

# Predict do modelo
tree_grid_pred_train = tree_best.predict(X_train)
tree_grid_pred_test = tree_best.predict(X_test)


#%% Plotando a árvore após o grid search

plt.figure(figsize=(20,10), dpi=600)
plot_tree(tree_best,
          feature_names=X.columns.tolist(),
          filled=True,
          node_ids=True)
plt.show()


#%% Avaliando o novo modelo (base de treino)

mse_train_tree_grid = mean_squared_error(y_train, tree_grid_pred_train)
mae_train_tree_grid = mean_absolute_error(y_train, tree_grid_pred_train)
r2_train_tree_grid = r2_score(y_train, tree_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_tree_grid):.1f}")
print(f"MAE: {mae_train_tree_grid:.1f}")
print(f"R²: {r2_train_tree_grid:.1%}")

#%% Avaliando o novo modelo (base de teste) Árvore de Regressão

mse_test_tree_grid = mean_squared_error(y_test, tree_grid_pred_test)
mae_test_tree_grid = mean_absolute_error(y_test, tree_grid_pred_test)
r2_test_tree_grid = r2_score(y_test, tree_grid_pred_test)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_tree_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_tree_grid):.1f}")
print(f"MAE: {mae_test_tree_grid:.1f}")
print(f"R²: {r2_test_tree_grid:.1%}")


#%% from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# Criando listas e dicionário para armazenar modelos, métricas e previsões
modelos_dst = []
rmse_dst = []
previsoes_dst = {}

# Calculando as métricas
mse_test_tree_grid = mean_squared_error(y_test, tree_grid_pred_test)  # Calcula MSE
r2_test_tree_grid = r2_score(y_test, tree_grid_pred_test)  # Calcula R²
rmse_test_tree_grid = np.sqrt(mse_test_tree_grid)  # Calcula RMSE corretamente

# Armazenando as informações
nome_modelo = 'Árvore de Regressão'
modelos_dst.append(nome_modelo)
rmse_dst.append(rmse_test_tree_grid)

# Exibindo os dados armazenados
print("\nDados Armazenados:")
print("Modelos:", modelos_dst)
print("RMSE:", rmse_dst)

# Exibindo as métricas
print("Avaliação do Modelo (Base de Teste) - Árvore de Regressão")
print(f"RMSE: {rmse_test_tree_grid:.1f}")
print(f"R²: {r2_test_tree_grid:.1%}")

#%% Importância das variáveis preditoras


import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


tree_features = pd.DataFrame({'features':X.columns.tolist(),
                              'importance':tree_best.feature_importances_}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(tree_features)

# Criar o gráfico de barras
plt.figure(figsize=(10, 6))
sns.barplot(x='importance', y='features', data=tree_features, palette='viridis')

# Adicionar rótulos e título
plt.xlabel('Importância da Variável')
plt.ylabel('Variáveis Preditoras')
plt.title('Importância das Variáveis Preditoras')

# Adicionar os valores de importância nas barras
for index, value in enumerate(tree_features['importance']):
    plt.text(value, index, f'{value:.1%}', va='center')

# Mostrar o gráfico
plt.tight_layout()
plt.show()


#%% Gráfico fitted values

# Valores preditos pelo modelo para as observações da amostra de teste
graph = pd.DataFrame({'dst': y_test,
                      'pred_tree': tree_grid_pred_test})

plt.figure(dpi=600)
sns.scatterplot(graph, x='dst', y='pred_tree', color='blue')
plt.title('Analisando as Previsões', fontsize=10)
plt.xlabel('Dst Observado', fontsize=10)
plt.ylabel('Dst Previsto pelo Modelo', fontsize=10)
plt.axline((25, 25), (max(dados['dst']), max(dados['dst'])), linewidth=1, color='grey')
plt.show()


#%%######################### Rondom Forest Regressor ################################
#                                                                             #
###############################################################################

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

#%% Iniciando o Grid Search

## Alguns hiperparâmetros do modelo

# n_estimators: qtde de árvores na floresta
# max_depth: profundidade máxima da árvore
# max_features: qtde de variáveis X consideradas na busca pelo melhor split
# min_samples_leaf: qtde mínima de observações para ser nó folha

# Vamos aplicar um Grid Search
param_grid_rf = {
    'n_estimators': [100, 500],
    'max_depth': [5, 10],
    'max_features': [3, 5, 7],
    'min_samples_leaf': [30, 50]
}

# Definindo o modelo
rf_grid = RandomForestRegressor(random_state=42)

# Treinar os modelos para o grid search
rf_grid_model = GridSearchCV(estimator = rf_grid, 
                             param_grid = param_grid_rf,
                             scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                             cv=5, verbose=2)

rf_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
rf_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
rf_best = rf_grid_model.best_estimator_

# Predict do modelo
rf_grid_pred_train = rf_best.predict(X_train)
rf_grid_pred_test = rf_best.predict(X_test)

 #%% Importância das variáveis preditoras

rf_features = pd.DataFrame({'features':X_train.columns.tolist(),
                            'importance':np.round(rf_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(rf_features)

#%% Avaliando a RF (base de treino)

mse_train_rf_grid = mean_squared_error(y_train, rf_grid_pred_train)
mae_train_rf_grid = mean_absolute_error(y_train, rf_grid_pred_train)
r2_train_rf_grid = r2_score(y_train, rf_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_rf_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_rf_grid):.1f}")
print(f"MAE: {mae_train_rf_grid:.1f}")
print(f"R²: {r2_train_rf_grid:.1%}")

# Nome das métricas e seus valores
metricas = {'R²': r2_train_rf_grid, 'RMSE': np.sqrt(mse_train_rf_grid), 'MSE': mse_train_rf_grid, 'MAE': mae_train_rf_grid}

# Ordenando as métricas do maior para o menor
metricas_sorted = dict(sorted(metricas.items(), key=lambda item: item[1], reverse=True))

# Extraindo os nomes e valores das métricas ordenadas
metric_names = list(metricas_sorted.keys())
metric_values = list(metricas_sorted.values())

# Criação do gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
sns.barplot(x=metric_values, y=metric_names, palette='viridis')

# Adicionando título e rótulos
plt.title("Resumo do Modelo", fontsize=16)
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores das barras
for i, v in enumerate(metric_values):
    plt.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=12)

# Exibindo o gráfico
plt.show()

#%% Avaliando a RF (base de teste)

mse_test_rf_grid = mean_squared_error(y_test, rf_grid_pred_test)
mae_test_rf_grid = mean_absolute_error(y_test, rf_grid_pred_test)
r2_test_rf_grid = r2_score(y_test, rf_grid_pred_test)
rmse_test_rf_grid = np.sqrt(mse_test_rf_grid)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_rf_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_rf_grid):.1f}")
print(f"MAE: {mae_test_rf_grid:.1f}")
print(f"R²: {r2_test_rf_grid:.1%}")

# Armazenando as informações
nome_modelo = 'Rondom Forest Regressor'
modelos_dst.append(nome_modelo)
rmse_dst.append(rmse_test_rf_grid)

# Nome das métricas e seus valores
metricas = {'R²': r2_test_rf_grid, 'RMSE': np.sqrt(mse_test_rf_grid), 'MSE': mse_test_rf_grid, 'MAE': mae_test_rf_grid}

# Ordenando as métricas do maior para o menor
metricas_sorted = dict(sorted(metricas.items(), key=lambda item: item[1], reverse=True))

# Extraindo os nomes e valores das métricas ordenadas
metric_names = list(metricas_sorted.keys())
metric_values = list(metricas_sorted.values())

# Criação do gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
sns.barplot(x=metric_values, y=metric_names, palette='viridis')

# Adicionando título e rótulos
plt.title("Rondom Forest Regressor", fontsize=16)
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores das barras
for i, v in enumerate(metric_values):
    plt.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=12)

# Exibindo o gráfico
plt.show()


#%%######################### Modelo XGBoost Regressor ################################
#                                                                             #
###############################################################################

#%%
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score

# Separando os dados em treino e teste (70% treino, 30% teste)
from sklearn.model_selection import train_test_split

xgb_model = xgb.XGBRegressor (test_size=0.3, random_state=42)

xgb_model.fit(X_train, y_train)


# Fazendo previsões
y_pred_train = xgb_model.predict(X_train)
y_pred_test = xgb_model.predict(X_test)

# Avaliação do modelo
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)

# Calculando RMSE manualmente
rmse_train = np.sqrt(mse_train)
rmse_test = np.sqrt(mse_test)

# Calculando MAE para os conjuntos de treino e teste
mae_train = mean_absolute_error(y_train, y_pred_train)
mae_test = mean_absolute_error(y_test, y_pred_test)

#%%
r2_train = r2_score(y_train, y_pred_train)

print("Avaliação do Modelo (Base de Treino):")
print(f"MSE: {mse_train:.1f}")
print(f"RMSE: {rmse_train:.1f}")
print(f"MAE: {mae_train:.1f}")
print(f"R²: {r2_train:.1%}")

# Nome das métricas e seus valores
metricas = {'R²': r2_train, 'RMSE': rmse_train, 'MSE': mse_train, 'MAE': mae_train}

# Ordenando as métricas do maior para o menor
metricas_sorted = dict(sorted(metricas.items(), key=lambda item: item[1], reverse=True))

# Extraindo os nomes e valores das métricas ordenadas
metric_names = list(metricas_sorted.keys())
metric_values = list(metricas_sorted.values())

# Criação do gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
sns.barplot(x=metric_values, y=metric_names, palette='viridis')

# Adicionando título e rótulos
plt.title("Resumo do Modelo", fontsize=16)
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores das barras
for i, v in enumerate(metric_values):
    plt.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=12)

# Exibindo o gráfico
plt.show()

#%%
r2_test = r2_score(y_test, y_pred_test)

print("\nAvaliação do Modelo (Base de Teste):")
print(f"MSE: {mse_test:.1f}")
print(f"RMSE: {rmse_test:.1f}")
print(f"MAE: {mae_test:.1f}")
print(f"R²: {r2_test:.1%}")

# Nome das métricas e seus valores
metricas = {'R²': r2_test, 'RMSE': rmse_test, 'MSE': mse_test, 'MAE': mae_test}

# Ordenando as métricas do maior para o menor
metricas_sorted = dict(sorted(metricas.items(), key=lambda item: item[1], reverse=True))

# Extraindo os nomes e valores das métricas ordenadas
metric_names = list(metricas_sorted.keys())
metric_values = list(metricas_sorted.values())

# Criação do gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
sns.barplot(x=metric_values, y=metric_names, palette='viridis')

# Adicionando título e rótulos
plt.title("Resumo do Modelo", fontsize=16)
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores das barras
for i, v in enumerate(metric_values):
    plt.text(v + 0.2, i, f"{v:.2f}", va='center', fontsize=12)

# Exibindo o gráfico
plt.show()

#%% Visualizar a importância das features
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title('Variáveis Impostantes', fontsize=16)
plt.ylabel('Variável', fontsize=12)
plt.grid(False)
plt.show()

#%%######################### XGBoost ##########################################
###############################################################################
#%% Iniciando o Grid Search

## Alguns hiperparâmetros do modelo
# n_estimators: qtde de árvores no modelo
# max_depth: profundidade máxima das árvores
# learning_rate: taxa de aprendizagem
# colsample_bytree: percentual de variáveis X subamostradas para cada árvore

# Vamos aplicar um Grid Search
param_grid_xgb = {
    'n_estimators': [100, 500, 700],
    'max_depth': [3, 5],
    'learning_rate': [0.001, 0.01, 0.1],
    'colsample_bytree': [0.5, 0.8],
}

# Identificar o algoritmo em uso
xgb_grid = XGBRegressor(random_state=100)

# Treinar os modelos para o grid search
xgb_grid_model = GridSearchCV(estimator = xgb_grid, 
                             param_grid = param_grid_xgb,
                             scoring='neg_mean_squared_error', # Atenção à metrica de avaliação!
                             cv=5, verbose=2)

xgb_grid_model.fit(X_train, y_train)

# Verificando os melhores parâmetros obtidos
xgb_grid_model.best_params_

# Gerando o modelo com os melhores hiperparâmetros
xgb_best = xgb_grid_model.best_estimator_

# Predict do modelo
xgb_grid_pred_train = xgb_best.predict(X_train)
xgb_grid_pred_test = xgb_best.predict(X_test)

#%% Importância das variáveis preditoras

xgb_features = pd.DataFrame({'features':X.columns.tolist(),
                             'importance':np.round(xgb_best.feature_importances_, 4)}).sort_values(by='importance', ascending=False).reset_index(drop=True)

print(xgb_features)

#%% Avaliando o XGB (base de treino)

mse_train_xgb_grid = mean_squared_error(y_train, xgb_grid_pred_train)
mae_train_xgb_grid = mean_absolute_error(y_train, xgb_grid_pred_train)
r2_train_xgb_grid = r2_score(y_train, xgb_grid_pred_train)

print("Avaliação do Modelo (Base de Treino)")
print(f"MSE: {mse_train_xgb_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_train_xgb_grid):.1f}")
print(f"MAE: {mae_train_xgb_grid:.1f}")
print(f"R²: {r2_train_xgb_grid:.1%}")

#%% Avaliando o XGB (base de teste)

mse_test_xgb_grid = mean_squared_error(y_test, xgb_grid_pred_test)
mae_test_xgb_grid = mean_absolute_error(y_test, xgb_grid_pred_test)
r2_test_xgb_grid = r2_score(y_test, xgb_grid_pred_test)
rmse_test_xg_grid = np.sqrt(mse_test_xgb_grid)

print("Avaliação do Modelo (Base de Teste)")
print(f"MSE: {mse_test_xgb_grid:.1f}")
print(f"RMSE: {np.sqrt(mse_test_xgb_grid):.1f}")
print(f"MAE: {mae_test_xgb_grid:.1f}")
print(f"R²: {r2_test_xgb_grid:.1%}")

# Armazenando as informações
nome_modelo = 'XGBoost Regressor'
modelos_dst.append(nome_modelo)
rmse_dst.append(rmse_test_xg_grid)


#%% 5. AVALIAÇÃO
###############################################################################
# Comparação dos modelos com base no MAPE
mape_comparison = pd.DataFrame({'Modelo': nome_modelo, 'RMSE': rmse_dst})
mape_comparison = mape_comparison.sort_values(by='RMSE', ascending=True).reset_index(drop=True)
print(mape_comparison)

# In[236]: Gráfico dos RMSE dos modelos
plt.figure(figsize=(10, 6))
plt.barh(mape_comparison['Modelo'], mape_comparison['RMSE'], color='skyblue')
plt.xlabel("RMSE")
plt.title("RMSE Comparação de Modelos")
plt.grid(True)
plt.show()

# In[237]: Selecionar o modelo com o menor MAPE
melhor_modelo = mape_comparison.loc[0, 'Modelo']
melhores_previsoes = previsoes_dst[melhor_modelo]

# In[238]: Criar gráfico comparando os valores reais e previstos do melhor modelo
plt.figure(figsize=(10, 6))
plt.plot(reaisenergia.index, reaisenergia, label='Valores Reais', color='blue')
plt.plot(reaisenergia.index, melhores_previsoes, label=f'Previsão - {melhor_modelo}', color='red')
plt.title(f'Valores Reais vs Previsão ({melhor_modelo})')
plt.xlabel('Data')
plt.ylabel('Valores')
plt.legend()
plt.grid(True)
plt.show()




