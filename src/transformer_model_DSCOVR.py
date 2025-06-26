# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 17:26:25 2025

@author: familia
"""

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import pickle
from sklearn.model_selection import train_test_split, \
    StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score
import tensorflow
import matplotlib.pyplot as plt
from tensorflow import keras
from keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Correção na variável df
dados = pd.read_pickle('df_prox_h_ds.pkl')

#%%#%% # 3. PREPARAÇÃO DOS DADOS - Separando as variáveis Y e X

# Função para engenharia de atributos
def dst_func_indicador(df):
    # Cópia do dataframe
    df_copy = df.copy()

    # Simple Moving Average (SMA)
    df_copy["SMA 15"] = df_copy["dst"].rolling(15).mean().shift(1)
    df_copy["SMA 60"] = df_copy["dst"].rolling(60).mean().shift(1)

    # Moving Standard Deviation (MSD) - Volatilidade
    df_copy["MSD 15"] = df_copy["dst"].rolling(15).std().shift(1)
    df_copy["MSD 60"] = df_copy["dst"].rolling(60).std().shift(1)

    # Rolling Window Estatísticas
    df_copy["Min 15"] = df_copy["dst"].rolling(15).min().shift(1)
    df_copy["Max 15"] = df_copy["dst"].rolling(15).max().shift(1)
    df_copy["Median 15"] = df_copy["dst"].rolling(15).median().shift(1)

    # Taxa de Variação
    df_copy["Taxa 15"] = df_copy["dst"] - df_copy["dst"].shift(15)

    # Interações Entre Variáveis
    df_copy["vel_mag"] = df_copy["vel"] * df_copy["mag"]

    return df_copy.dropna()

#%% Aplicar a engenharia de atributos nos dados reais
df_real = dst_func_indicador(dados)

# Selecionar as variáveis preditoras (X) e a variável alvo (y)
X_reais = df_real[['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15', 'Taxa 15', 'vel_mag']]
y_reais = df_real[['prox_h']]

#%%# Padronização (usando o mesmo StandardScaler 
from sklearn.preprocessing import StandardScaler

# Carregar o scaler treinado ou treinar um novo (preferencialmente, utilize o mesmo scaler)
scaler = StandardScaler()
X_reais_sc = scaler.fit_transform(X_reais)  # Ou use scaler.transform() se tiver o scaler salvo

#%%
import numpy as np

def dst_ajusta_formato_dados(X_s, y_s, lag):
    # Verifica se o comprimento de X_s é igual ao de y_s
    if len(X_s) != len(y_s):
        print("Warning: X_s e y_s têm comprimentos diferentes.")

    # Inicializa a lista X_train; Construção do X_train com janelas deslizantes;
    X_train = []

    # Itera sobre as variáveis (colunas) de X_s
    for variable in range(X_s.shape[1]):
        X = []
        for i in range(lag, X_s.shape[0]):
            X.append(X_s[i - lag:i, variable])
        X_train.append(X)

    # Converte X_train para um array numpy e ajusta os eixos
    X_train = np.array(X_train)  # Corrigido
    X_train = np.swapaxes(np.swapaxes(X_train, 0, 1), 1, 2)

    # Inicializa a lista y_train
    y_train = []

    # Ajusta y_train com base no lag
    for i in range(lag, y_s.shape[0]):
        y_train.append(y_s[i].reshape(1, -1))  # Corrigido

    # Concatena y_train corretamente
    y_train = np.concatenate(y_train, axis=0)

    return X_train, y_train

#%% 
lag = 15  # Exemplo de 24 horas de lag

# Ajuste os dados para o formato de séries temporais
X_reais_final, y_reais_final = dst_ajusta_formato_dados(X_reais_sc, y_reais.values, lag)

#%% 
# Carregar o modelo salvo

with open('modelo_dst.pkl', 'rb') as file:
    modelo_carregado = pickle.load(file)
    
# Fazer previsões
previsoes = modelo_carregado.predict(X_reais_final)

#%% Visualizar: Gráfico Real vs. Predito
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
plt.plot(y_reais_final[:200], label='Real')
plt.plot(previsoes[:200], label='Previsto')
#plt.title('Comparação entre valores reais e previstos (200 primeiros pontos)')
plt.xlabel('Amostras')
plt.ylabel('DST')
plt.legend()
plt.tight_layout()
plt.show()

#%% Gráfico de dispersão

min_len = min(len(y_reais_final), len(previsoes))
y_true = y_reais_final.flatten()[:min_len]
y_pred_ = previsoes.flatten()[:min_len]

ax = sns.regplot(x=y_true, y=y_pred_, scatter_kws={'alpha': 0.3})

plt.plot([y_reais_final.min(), y_reais_final.max()], [y_reais_final.min(), y_reais_final.max()], '--r')
#plt.title('Diagrama de Dispersão: Real vs Predito')
plt.xlabel('Valores Reais')
plt.ylabel('Valores Preditos')

# Ajustando a estética do gráfico
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

plt.tight_layout()
plt.show()

#%% Garantindo 1D
y_true = y_reais_final.flatten()[:200]
y_pred_ = previsoes.flatten()[:200]

plt.figure(figsize=(10, 6))
ax = sns.lineplot(x=range(len(y_true)), y=y_true, label='Real')
sns.lineplot(x=range(len(y_pred_)), y=y_pred_, label='Predito')
plt.xlabel('Amostras')
plt.ylabel('Valor')
#plt.title('Comparação: Valores Reais vs Preditos (200 amostras)')
plt.legend()

# Ajustando a estética do gráfico
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

plt.tight_layout()
plt.show()

#%% Visualização dos Resultados
import matplotlib.pyplot as plt

# Definindo cores personalizadas
cores = {
    'laranja': (255/255, 127/255, 14/255),  # Laranja
    'azul': (31/255, 119/255, 180/255),      # Azul
}

# Criando figura com fundo branco
fig, ax = plt.subplots(figsize=(8, 6))
fig.patch.set_facecolor('white')  # Garante fundo branco ao salvar

# Plotando os dados
ax.plot(y_reais_final, label='Valores reais', 
        color=cores['azul'], linewidth=2)
ax.plot(previsoes, label='Previsões', 
        color=cores['laranja'], linewidth=2, linestyle='--')

# Configurando eixos
ax.set_xlabel('Amostras', fontsize=12)
ax.set_ylabel('Dst', fontsize=12)

# Configurando legenda
ax.legend(frameon=False, fontsize=11)

# Ajustando bordas
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Removendo grid
ax.grid(False)

# Ajuste final
plt.tight_layout()
plt.show()

#%% Exibir os resultados
print("Previsões:", previsoes)

# Calcula as métricas
r2_tft = metrics.r2_score(y_reais_final, previsoes)
mae_tft = metrics.mean_absolute_error(y_reais_final, previsoes)
mse_tft = metrics.mean_squared_error(y_reais_final, previsoes)
rmse_tft = np.sqrt(mse_tft)

# Exibindo as métricas
print("Resumo do Modelo:")
print(f"R²: {r2_tft:.1%}")
print(f"MAE: {mae_tft:.2f}")
print(f"MSE: {mse_tft:.2f}")
print(f"RMSE: {rmse_tft:.2f}")

# Criando um DataFrame com as métricas (substitui o dicionário original)
df_metricas = pd.DataFrame({
    'Métrica': ['R²', 'MAE', 'MSE', 'RMSE'],
    'Valor': [r2_tft, mae_tft, mse_tft, rmse_tft]
})

# Ordenando as métricas do maior para o menor
df_sorted = df_metricas.sort_values('Valor', ascending=False)

# Criação do gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Valor', y='Métrica', data=df_sorted, palette='viridis')

# Adicionando título e rótulos
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores das barras
for i, v in enumerate(df_sorted['Valor']):
    metrica = df_sorted.iloc[i]['Métrica']
    if metrica == 'R²':  # Se for R², formata como percentual
        ax.text(v + 0.01, i, f"{v:.1%}", va='center', color='black', fontweight='bold', fontsize=12)
    else:
        ax.text(v + 0.01, i, f"{v:.2f}", va='center', color='black', fontweight='bold', fontsize=12)

# Removendo bordas superior e direita
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

plt.savefig('importancia_var.png', dpi=100, bbox_inches='tight', pad_inches=0)
# Exibindo o gráfico
plt.show()



