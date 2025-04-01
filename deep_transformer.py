# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:32:09 2025

@author: familia
"""
#%% Importar os pacotes

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from sklearn.preprocessing import RobustScaler, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Importar a base (já tratada)
data = pd.read_pickle('df_prox_h.pkl')

#%% Passo 1: Pré-processamento dos Dados

import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Normalizar entre [-1, 1] (importante para DST, que pode ser negativo)
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(data)

# Separar features (X) e target (y = prox_h)
target_index = data.columns.get_loc('prox_h')  # Índice da coluna 'prox_h'
X_data = np.delete(scaled_data, target_index, axis=1)  # Todas as colunas exceto 'prox_h'
y_data = scaled_data[:, target_index]  

# Criar sequências temporais (janela deslizante)
def create_sequences(X, y, window_size):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])  # Janela de 'window_size' steps
        y_seq.append(y[i+window_size])    # Target (próximo valor de 'prox_h')
    return np.array(X_seq), np.array(y_seq)

window_size = 24  # Janela de 24 horas (ajuste conforme necessidade)
X, y = create_sequences(X_data, y_data, window_size)

#%% Engenharia de atributos
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

# Aplicar engenharia de atributos
df = dst_func_indicador(data)

#%% Separação treino, teste e validação
def get_train_test_val(data, test_ratio=0.1, val_ratio=0.05):
    test_size = int(len(data) * test_ratio)
    val_size = int(len(data) * val_ratio)

    test = data.tail(test_size).reset_index(drop=True)
    interim = data.iloc[:-test_size]
    val = interim.tail(val_size).reset_index(drop=True)
    train = interim.iloc[:-val_size]

    return train, test, val

# Chamar a função corretamente
train, test, val = get_train_test_val(df)

# Definir X (variáveis preditoras) e y (alvo) para cada conjunto de dados
features = ['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15', 'Taxa 15', 'vel_mag']

X_train = train[features].values
y_train = train['prox_h'].values

X_valid = val[features].values
y_valid = val['prox_h'].values

X_test = test[features].values
y_test = test['prox_h'].values

# Redimensionar para formato 3D (samples, timesteps, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_valid = X_valid.reshape((X_valid.shape[0], 1, X_valid.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(f"Treino: {len(train)}, Validação: {len(val)}, Teste: {len(test)}")
print("Shape de X_train:", X_train.shape)

#%% Modelo LSTM Multivariado - Arquitetura com Regularização
from tensorflow.keras import models, layers, regularizers

n_features = X_train.shape[2]  # Número de features (9)

model = models.Sequential([
    layers.LSTM(64, 
                activation='tanh',
                return_sequences=True,
                input_shape=(1, n_features),  # Alterado para (1, n_features)
                kernel_regularizer=regularizers.l2(0.01)),
    layers.Dropout(0.3),
    layers.LSTM(32, activation='tanh'),
    layers.Dropout(0.3),
    layers.Dense(1)  # Saída única (valor previsto de 'prox_h')
])

model.compile(
    optimizer='adam',
    loss='mse',  # Mean Squared Error para regressão
    metrics=['mae']  # Mean Absolute Error
)

model.summary()

#%% Treinamento com Callbacks

from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

callbacks = [
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    callbacks=callbacks,
    verbose=1
)

#%% Avaliação e Previsões

import matplotlib.pyplot as plt

# Gráfico de Loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training vs Validation Loss')
plt.legend()
plt.show()

#%% Prever no conjunto de teste
# 1. Primeiro, vamos criar um scaler específico para o target (prox_h)
target_scaler = MinMaxScaler(feature_range=(-1, 1))
target_scaler.fit(data[['prox_h']])  # Ajuste apenas na coluna target

# 2. Previsões
y_pred = model.predict(X_test)

# 3. Reverter a normalização APENAS para o target
y_pred_real = target_scaler.inverse_transform(y_pred)
y_test_real = target_scaler.inverse_transform(y_test.reshape(-1, 1))

# 4. Plotar resultados (exemplo)
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 6))
plt.plot(y_test_real, label='Valores Reais')
plt.plot(y_pred_real, label='Previsões', alpha=0.7)
plt.legend()
plt.title('Comparação: Valores Reais vs Previsões')
plt.xlabel('Amostras')
plt.ylabel('prox_h')
plt.show()
#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import pandas as pd  # Adicionei esta importação que estava faltando


# Calcula as métricas
r2_tft = metrics.r2_score(y_test_real, y_pred_real)
mae_tft = metrics.mean_absolute_error(y_test_real, y_pred_real)
mse_tft = metrics.mean_squared_error(y_test_real, y_pred_real)
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

#%% OTIMIZAÇÃO DO MODELO COM REGULARIZAÇÃO AVANÇADA
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configurações avançadas
optimizer = Adam(learning_rate=0.001, clipvalue=0.5)
early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=10, min_lr=1e-6)

# Arquitetura melhorada
model = models.Sequential([
    layers.LSTM(128, 
                activation='tanh',
                return_sequences=True,
                input_shape=(window_size, n_features),
                kernel_regularizer=regularizers.l1_l2(l1=0.01, l2=0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
    layers.LSTM(64, activation='tanh', return_sequences=True),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.LSTM(32, activation='tanh'),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer=optimizer, loss='huber', metrics=['mae'])

# Treinamento com callbacks
history = model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=64,
    validation_data=(X_valid, y_valid),
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

#%% AVALIAÇÃO APRIMORADA
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"\n⚡ Métricas Finais ⚡")
    print(f"R²: {r2:.3%}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    return y_pred

y_pred = evaluate_model(model, X_test, y_test)

#%%
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd  

# Supondo que y_test e y_pred já estejam definidos
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Exibindo as métricas
print("Resumo do Modelo:")
print(f"R²: {r2:.2f}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

# Criando um DataFrame com as métricas
df_metricas = pd.DataFrame({
    'Métrica': ['R²', 'MAE', 'MSE', 'RMSE'],
    'Valor': [r2, mae, mse, rmse]
})

# Ordenando as métricas do maior para o menor
df_sorted = df_metricas.sort_values('Valor', ascending=False)

# Criando o gráfico de barras com a paleta 'viridis'
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Valor', y='Métrica', data=df_sorted, palette='viridis')

# Adicionando título e rótulos
plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)

# Exibindo os valores nas barras
for i, v in enumerate(df_sorted['Valor']):
    metrica = df_sorted.iloc[i]['Métrica']
    texto = f"{v:.2f}" if metrica != 'R²' else f"{v:.1%}"  # Formata R² como percentual
    ax.text(v + 0.01, i, texto, va='center', color='black', fontweight='bold', fontsize=12)

# Ajustando a estética do gráfico
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

# Salvando o gráfico
plt.savefig('importancia_var.png', dpi=100, bbox_inches='tight', pad_inches=0)

# Exibindo o gráfico
plt.show()

#%% VISUALIZAÇÃO AVANÇADA
import matplotlib.pyplot as plt
import seaborn as sns

# Gráfico de séries temporais
plt.figure(figsize=(8, 6))
ax = sns.lineplot(x=range(len(y_test[:200])), y=y_test[:200], label='Real')
sns.lineplot(x=range(len(y_pred[:200])), y=y_pred[:200].flatten(), label='Predito')
#plt.title('Comparação: Valores Reais vs Preditos (200 amostras)')
plt.xlabel('Tempo')
plt.ylabel('DST')

# Ajustando a estética do gráfico
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

plt.tight_layout()
plt.show()
#%% Gráfico de dispersão

plt.figure(figsize=(8, 6))
ax = sns.regplot(x=y_test.flatten(), y=y_pred.flatten(), scatter_kws={'alpha':0.3})
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], '--r')
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


