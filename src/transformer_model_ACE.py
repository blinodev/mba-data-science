# -*- coding: utf-8 -*-
"""
Created on Mon Apr 14 19:20:20 2025

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
dados = pd.read_pickle('df_prox_h.pkl')

#%% Função para engenharia de atributos

def dst_func_indicador(df):

    # Cópia do dataframe
    df_copy = dados.copy()

    # Simple Moving Average (SMA)
    df_copy["SMA 15"] = df_copy[["dst"]].rolling(15).mean().shift(1)
    df_copy["SMA 60"] = df_copy[["dst"]].rolling(60).mean().shift(1)

    # Moving Standard Deviation (MSD) - Volatilidade
    df_copy["MSD 15"] = df_copy["prox_h"].rolling(15).std().shift(1)
    df_copy["MSD 60"] = df_copy["prox_h"].rolling(60).std().shift(1)

    # Rolling Window Estatísticas: Além de médias e desvios, calcular mínimos, máximos e medianas pode trazer insights adicionais:
    df_copy["Min 15"] = df_copy["dst"].rolling(15).min().shift(1)
    df_copy["Max 15"] = df_copy["dst"].rolling(15).max().shift(1)
    df_copy["Median 15"] = df_copy["dst"].rolling(15).median().shift(1)

# Taxa de Variação : Para capturar tendências de alta ou queda:

    df_copy["Taxa 15"] = df_copy["dst"] - df_copy["dst"].shift(15)

# Interações Entre Variáveis: Multiplicação de variáveis para capturar relações não lineares:
    df_copy["vel_mag"] = df_copy["vel"] * df_copy["mag"]

    return df_copy.dropna()

#%% Engenharia de atributos
df = dst_func_indicador(dados)

#%% Separação treino, teste e validação
def get_train_test_val(data, test_ratio=0.1, val_ratio=0.05):
    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)

    test = data.tail(test_size).reset_index(drop=True)
    interim = data.iloc[:-test_size]
    val = interim.tail(val_size).reset_index(drop=True)
    train = interim.iloc[:-val_size]

    return train, test, val

print(df.columns)

#%% Chamar a função corretamente
train, test, val = get_train_test_val(df)

# Definir X (variáveis preditoras) e y (alvo) para cada conjunto de dados
x_treino = train[['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15',
'Taxa 15', 'vel_mag']]
y_treino = train[['prox_h']]

x_valid = val[['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15',
'Taxa 15', 'vel_mag']]
y_valid = val[['prox_h']]

x_teste = test[['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15',
'Taxa 15', 'vel_mag']]
y_teste = test[['prox_h']]

# Agora sim, podemos imprimir os tamanhos corretamente
print(f"Treino: {len(train)}, Validação: {len(val)}, Teste: {len(test)}")

#%% Padronização
# Cria o padronizador
sc = StandardScaler()
# Fit e transform nos dados de treino
x_treino_sc = sc.fit_transform(x_treino)
# Transform nos dados de validação
x_valid_sc = sc.transform(x_valid)
# Transform nos dados de teste
x_teste_sc = sc.transform(x_teste)

#%%
import numpy as np

def dsa_ajusta_formato_dados(X_s, y_s, lag):
    # Verifica se o comprimento de X_s é igual ao de y_s
    if len(X_s) != len(y_s):
        print("Warning: X_s e y_s têm comprimentos diferentes.")

    # Inicializa a lista X_train
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

#%% Valor do Lag

import numpy as np
lag = 24

# Aplica a função nos dados de treino
x_treino_final, y_treino_final = dsa_ajusta_formato_dados(x_treino_sc, y_treino.values, lag)

# Aplica a função nos dados de validação
x_valid_final, y_valid_final = dsa_ajusta_formato_dados(x_valid_sc, y_valid.values, lag)

# Aplica a função nos dados de teste
x_teste_final, y_teste_final = dsa_ajusta_formato_dados(x_teste_sc, y_teste.values, lag)

print(f"Shape dos dados de treino: {x_treino_final.shape}")
print(f"Shape dos dados de validação: {x_valid_final.shape}")
print(f"Shape dos dados de teste: {x_teste_final.shape}")

#%% Construção do Modelo Temporal Fusion Transformer
# Função do transformer encoder
def dst_transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout = 0):

    # Normaliza as entradas com camadas de normalização
    x = layers.LayerNormalization(epsilon = 1e-6)(inputs)

    # Aplica a atenção multi-cabeça nas entradas
    x = layers.MultiHeadAttention(key_dim = head_size, num_heads = num_heads, dropout = dropout)(x, x, x)

    # Aplica Dropout na saída da camada de atenção
    x = layers.Dropout(dropout)(x)

    # Adiciona as entradas iniciais como conexão residual
    res = x + inputs

    # Normaliza a soma da conexão residual
    x = layers.LayerNormalization(epsilon = 1e-6)(res)

    # Aplica uma camada convolucional com ativação ReLU
    x = layers.Conv1D(filters = ff_dim, kernel_size = 1, activation = "relu")(x)

    # Aplica Dropout após a camada convolucional
    x = layers.Dropout(dropout)(x)

    # Aplica uma segunda camada convolucional
    x = layers.Conv1D(filters = inputs.shape[-1], kernel_size = 1)(x)

    # Retorna a soma da segunda camada convolucional com a conexão residual
    return x + res

#%% Função de criação do modelo
def dst_cria_modelo(input_shape,
                    head_size,
                    num_heads,
                    ff_dim,
                    num_transformer_blocks,
                    mlp_units,
                    dropout = 0,
                    mlp_dropout = 0):

    # Define a entrada do modelo com a forma especificada
    inputs = keras.Input(shape = input_shape)

    # Inicializa a entrada do modelo na variável x
    x = inputs

    # Adiciona uma camada LSTM com 10 unidades e retorna sequências
    x = layers.LSTM(10, return_sequences = True)(x)

    # Adiciona blocos de transformer ao modelo
    for _ in range(num_transformer_blocks):
        x = dst_transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    # Adiciona uma camada GRU com 100 unidades e não retorna sequências
    x = layers.GRU(100, return_sequences = False)(x)

    # Adiciona uma camada Dropout com a taxa especificada
    x = layers.Dropout(mlp_dropout)(x)

    # Adiciona uma camada densa com unidades especificadas e ativação ReLU
    x = layers.Dense(mlp_units, activation = "relu")(x)

    # Define a camada de saída com 1 unidade (saída do modelo)
    outputs = layers.Dense(1)(x)

    # Retorna o modelo criado com as entradas e saídas definidas
    return keras.Model(inputs, outputs)

#%% Cria o modelo
input_shape = x_treino_final.shape[1:]

modelo_dst = dst_cria_modelo(input_shape,
                             head_size=32,
                             num_heads=2,
                             ff_dim=8,
                             num_transformer_blocks=2,
                             mlp_units=256,
                             dropout=0.3,
                             mlp_dropout=0.5)


#%% Compila o modelo
modelo_dst.compile(loss = "mean_squared_error", optimizer = keras.optimizers.Adam())

# Confirme qual modelo você está usando:
modelo_dst.summary()


callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss",
                                  patience=5,
                                  restore_best_weights=True,
                                   mode="min")
]

#%%
history = modelo_dst.fit(x_treino_final,
                         y_treino_final,
                         validation_data=(x_valid_final, y_valid_final),
                         epochs=20,
                         batch_size=128,
                         callbacks=callbacks)


# Plotando as perdas de treino e validação
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(history.history['loss'], label='Loss')
ax.plot(history.history['val_loss'], label='Validation Loss')
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
ax.grid(False)  # Adicionando a grade
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)
plt.show()

#%% #%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd  # Adicionei esta importação que estava faltando

# A previsões e o valor real de y_teste_final
pred = modelo_dst.predict(x_teste_final)

# Calcula as métricas
r2_tft = metrics.r2_score(y_teste_final, pred)
mae_tft = metrics.mean_absolute_error(y_teste_final, pred)
mse_tft = metrics.mean_squared_error(y_teste_final, pred)
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
ax.plot(y_teste_final, label='Valores reais', 
        color=cores['azul'], linewidth=2)
ax.plot(pred, label='Previsões', 
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

#%% Gráfico de dispersão

min_len = min(len(y_teste_final), len(pred))
y_true = y_teste_final.flatten()[:min_len]
y_pred_ = pred.flatten()[:min_len]

ax = sns.regplot(x=y_true, y=y_pred_, scatter_kws={'alpha': 0.3})

plt.plot([y_teste_final.min(), y_teste_final.max()], [y_teste_final.min(), y_teste_final.max()], '--r')
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
y_true = y_teste_final.flatten()[:200]
y_pred_ = pred.flatten()[:200]

plt.figure(figsize=(8, 6))
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


#%% Salvar o modelo final em um arquivo usando Pickle
with open('modelo_dst_lag24.pkl', 'wb') as file:
    pickle.dump(modelo_dst, file)
    
#%%Carregar o modelo salvo

with open('modelo_camp_dst.pkl', 'rb') as file:
    modelo_carregado = pickle.load(file)

# Fazer previsões
previsoes = modelo_carregado.predict(X_reais_final)
