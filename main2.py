import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
sys.path.append(os.path.abspath("../src")) 
from src.dados.engenharia import dst_func_indicador
from src.preprocessamento.formatadores import padroniza_dados, dsa_ajusta_formato_dados
from src.modelos.transformer import dst_cria_modelo
from src.metricas import avalia_modelo
import pandas as pd

# 1. Carrega dados
dados = pd.read_pickle('data/df_prox_h.pkl')
df = dst_func_indicador(dados)

# 2. Divide conjuntos
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.1, shuffle=False)
train, val = train_test_split(train, test_size=0.05, shuffle=False)

# 3. Seleciona features
features = ['SMA 15', 'SMA 60', 'MSD 15', 'MSD 60', 'Min 15', 'Max 15', 'Median 15', 'Taxa 15', 'vel_mag']
target = 'prox_h'

x_treino, y_treino = train[features], train[[target]]
x_val, y_val = val[features], val[[target]]
x_teste, y_teste = test[features], test[[target]]

# 4. Padroniza
x_treino_sc, x_val_sc, x_teste_sc = padroniza_dados(x_treino, x_val, x_teste)

# 5. Ajusta formato para transformer
lag = 24
x_treino_final, y_treino_final = dsa_ajusta_formato_dados(x_treino_sc, y_treino.values, lag)
x_val_final, y_val_final = dsa_ajusta_formato_dados(x_val_sc, y_val.values, lag)
x_teste_final, y_teste_final = dsa_ajusta_formato_dados(x_teste_sc, y_teste.values, lag)

# 6. Cria modelo
modelo = dst_cria_modelo(input_shape=x_treino_final.shape[1:],
                         head_size=32,
                         num_heads=2,
                         ff_dim=8,
                         num_transformer_blocks=2,
                         mlp_units=256,
                         dropout=0.3,
                         mlp_dropout=0.5)

modelo.compile(loss='mean_squared_error', optimizer='adam')

modelo.fit(x_treino_final, y_treino_final,
           validation_data=(x_val_final, y_val_final),
           epochs=20, batch_size=128)

# 7. Avaliação
y_pred = modelo.predict(x_teste_final)
metricas = avalia_modelo(y_teste_final, y_pred)
print(metricas)
