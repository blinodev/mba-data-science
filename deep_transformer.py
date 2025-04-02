# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 13:32:09 2025

@author: Blino
"""
#%%#############################################################################
#             Redes Neurais RNNs - Temporal Fusion transformer              #
#                        CARREGAMENTO DA BASE DE DADOS                      #
#############################################################################

#%% Importar os pacotes

# 1. Carregar e preparar dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.preprocessing import RobustScaler
from tensorflow.keras import layers, models
import tensorflow as tf

data = pd.read_pickle('df_prox_h.pkl')

#%% 2. Engenharia de atributos (corrigida)
def dst_func_indicador(df):
    df_copy = df.copy()
    
    # Features temporais
    for window in [15, 60]:
        df_copy[f"SMA_{window}"] = df_copy["dst"].rolling(window).mean().shift(1)
        df_copy[f"MSD_{window}"] = df_copy["dst"].rolling(window).std().shift(1)
    
    # Estatísticas adicionais
    df_copy["Min_15"] = df_copy["dst"].rolling(15).min().shift(1)
    df_copy["Max_15"] = df_copy["dst"].rolling(15).max().shift(1)
    df_copy["Median_15"] = df_copy["dst"].rolling(15).median().shift(1)
    df_copy["Taxa_15"] = df_copy["dst"].diff(15)
    
    # Interações entre variáveis
    df_copy["vel_mag"] = df_copy["vel"] * df_copy["mag"]
    df_copy["bz_negative"] = np.where(df_copy['bz_gsm'] < 0, df_copy['bz_gsm'], 0)
    
    return df_copy.dropna()

df = dst_func_indicador(data)

#%% 3. Seleção de features e target
features = ['SMA_15', 'SMA_60', 'MSD_15', 'MSD_60', 'Min_15', 'Max_15', 
            'Median_15', 'Taxa_15', 'vel_mag', 'bz_negative', 'bz_gsm', 
            'theta_gsm', 'mag', 'vel', 'temp', 'med']
target = 'prox_h'

#%% 4. Divisão dos dados (temporal)
def temporal_split(data, test_size=0.15, val_size=0.15):
    n = len(data)
    test_end = n
    test_start = n - int(n * test_size)
    val_end = test_start
    val_start = val_end - int(n * val_size)
    
    train = data.iloc[:val_start]
    val = data.iloc[val_start:val_end]
    test = data.iloc[test_start:test_end]
    
    return train, val, test

train, val, test = temporal_split(df)

# 5. Normalização (RobustScaler para lidar com outliers)
scaler_X = RobustScaler()
scaler_y = RobustScaler()

X_train = scaler_X.fit_transform(train[features])
y_train = scaler_y.fit_transform(train[[target]])

X_val = scaler_X.transform(val[features])
y_val = scaler_y.transform(val[[target]])

X_test = scaler_X.transform(test[features])
y_test = scaler_y.transform(test[[target]])

#%% 6. Criação de sequências temporais
def create_sequences(X, y, window_size=24):
    X_seq, y_seq = [], []
    for i in range(len(X) - window_size):
        X_seq.append(X[i:i+window_size])
        y_seq.append(y[i+window_size])
    return np.array(X_seq), np.array(y_seq)

X_train_seq, y_train_seq = create_sequences(X_train, y_train)
X_val_seq, y_val_seq = create_sequences(X_val, y_val)
X_test_seq, y_test_seq = create_sequences(X_test, y_test)

#%% 7. Arquitetura TFT otimizada
def build_tft_model(input_shape):
    inputs = layers.Input(shape=input_shape)
    
    # Camada de processamento temporal
    lstm_out = layers.LSTM(128, return_sequences=True, 
                          kernel_regularizer=tf.keras.regularizers.l2(0.01))(inputs)
    lstm_out = layers.LayerNormalization()(lstm_out)
    
    # Mecanismo de atenção
    attention = layers.MultiHeadAttention(
        num_heads=8,
        key_dim=64,
        dropout=0.1
    )(lstm_out, lstm_out)
    attention = layers.LayerNormalization()(attention + lstm_out)
    
    # Camada de saída
    output = layers.LSTM(64)(attention)
    output = layers.Dropout(0.2)(output)
    output = layers.Dense(1)(output)
    
    return models.Model(inputs=inputs, outputs=output)

model = build_tft_model((X_train_seq.shape[1], X_train_seq.shape[2]))
model.summary()

#%% 8. Compilação e treinamento
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='huber',  # Mais robusto a outliers
    metrics=['mae']
)

history = model.fit(
    X_train_seq, y_train_seq,
    validation_data=(X_val_seq, y_val_seq),
    epochs=20,
    batch_size=64,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)

#%% 9. Avaliação e Visualização
y_pred = model.predict(X_test_seq)
y_test_orig = scaler_y.inverse_transform(y_test_seq.reshape(-1, 1))
y_pred_orig = scaler_y.inverse_transform(y_pred.reshape(-1, 1))

# Criar figura e eixos
fig, ax = plt.subplots(figsize=(8, 6))

# Plotar dados
ax.plot(y_test_orig[:500], label='Valores Reais', color='blue')
ax.plot(y_pred_orig[:500], label='Previsões', color='orange', alpha=0.7)

# Configurações do gráfico
ax.set_xlabel('Amostras')
ax.set_ylabel('prox_h')
ax.legend()

# Configurações de borda e grid
ax.grid(False)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

plt.show()

#%% 10. Gráfico de Loss durante o treinamento
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Remover bordas desnecessárias
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.grid(False)
plt.show()

#%% 11. Métricas de avaliação
# Calcula as métricas
r2_tft = metrics.r2_score(y_test_orig, y_pred_orig)
mae_tft = metrics.mean_absolute_error(y_test_orig, y_pred_orig)
mse_tft = metrics.mean_squared_error(y_test_orig, y_pred_orig)
rmse_tft = np.sqrt(mse_tft)

# Exibindo as métricas
print("Resumo do Modelo:")
print(f"R²: {r2_tft:.1%}")
print(f"MAE: {mae_tft:.2f}")
print(f"MSE: {mse_tft:.2f}")
print(f"RMSE: {rmse_tft:.2f}")

# Criando DataFrame com as métricas
metricas = {'R²': r2_tft, 'MAE': mae_tft, 'MSE': mse_tft, 'RMSE': rmse_tft}
df_metricas = pd.DataFrame(list(metricas.items()), columns=['Métrica', 'Valor'])

# Ordenando as métricas do maior para o menor
df_sorted = df_metricas.sort_values('Valor', ascending=False)

# Criação do gráfico de barras
plt.figure(figsize=(8, 6))
ax = sns.barplot(x='Valor', y='Métrica', data=df_sorted, palette='viridis')

# Adicionando valores nas barras
for i, v in enumerate(df_sorted['Valor']):
    metrica = df_sorted.iloc[i]['Métrica']
    if metrica == 'R²':  # Se for R², formata como percentual
        plt.text(v + 0.2, i, f"{v:.1%}", va='center', color='black', fontweight='bold', fontsize=12)
    else:
        plt.text(v + 0.2, i, f"{v:.2f}", va='center', color='black', fontweight='bold', fontsize=12)

# Configurações de borda
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)    

plt.xlabel("Valores", fontsize=12)
plt.ylabel("Métricas", fontsize=12)
plt.show()