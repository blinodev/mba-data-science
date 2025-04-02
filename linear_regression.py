# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:27:43 2025

@author: Blino
"""

#%% Importando os pacotes
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statstests.process import stepwise
from statstests.tests import shapiro_francia
from scipy.stats import boxcox
from scipy.stats import norm
from scipy import stats


# Importar a base (já tratada)
dados = pd.read_pickle('df_final.pkl')

#%% 1. COMPREENSÃO DO PROBLEMA 
dados.info()
dados.head()

#%% 2. COMPREENSÃO DOS DADOS - Verificar valores faltantes
def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

relatorio_missing(dados)

#%% 3. PREPARAÇÃO DOS DADOS - Separando as variáveis Y e X

X = dados.drop(columns=['dst'])
y = dados['dst']

#%%  📌 Separando as amostras de treino e teste

# Vamos escolher 70% das observações para treino e 30% para teste
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42)


# 4 - MODELAGEM
#%%
#############################################################################
#                        REGRESSÃO NÃO LINEAR MULTIPLA                      #
#                        CARREGAMENTO DA BASE DE DADOS                      #
#############################################################################

# 📌 Modelo Regressão Não Linear (MQO)
X_train_mqo = sm.add_constant(X_train)
X_test_mqo = sm.add_constant(X_test)


#%% Fazer previsões

y_pred_mqo = modelo_mqo.predict(sm.add_constant(X_test))

# Avaliar o modelo
r2 = r2_score(y_test, y_pred_mqo)
mse = mean_squared_error(y_test, y_pred_mqo)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_mqo))
mae = mean_absolute_error(y_test, y_pred_mqo)

# Exibindo resultados
print("Resumo do Modelo:")
print(f"R²: {r2:.1%}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

joblib.dump(modelo_mqo, "modelo_mqo.pkl")

#%% Criar DataFrame para métricas
metrics = ['R²', 'MSE', 'RMSE', 'MAE']
values = [r2, mse, rmse, mae]
data = {'Métrica': metrics, 'Valor': values}
df1 = pd.DataFrame(data)

# Ordenar o DataFrame (opcional, mas recomendado)
df_sorted = df1.sort_values('Valor', ascending=False)  # Ordena do menor para o maior (para barras horizontais)

# Configurar o gráfico
plt.figure(figsize=(8, 6))
ax = sns.barplot(y='Métrica', x='Valor', data=df_sorted, palette='viridis', hue='Métrica')

# Adicionar anotações nas barras
for i, p in enumerate(ax.patches):
    metrica = df_sorted.iloc[i]['Métrica']  # Obtém a métrica associada à barra
    if metrica == 'R²':  # Se for R², formata como percentual
        plt.annotate(f"{p.get_width():.1%}", (p.get_width() * 1.005, p.get_y() + p.get_height() / 2), color='black', fontweight='bold', va='center', fontsize=12)
    else:
        plt.annotate(f"{p.get_width():.2f}", (p.get_width() * 1.005, p.get_y() + p.get_height() / 2), color='black', fontweight='bold', va='center', fontsize=12)

# Configurações adicionais do gráfico
plt.ylabel('Métrica')
plt.xlabel('Valor')
plt.tight_layout()

# Removendo bordas superior e direita
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_linewidth(1.5)
ax.spines['left'].set_linewidth(1.5)

# Salvar e exibir o gráfico
plt.savefig('desemp_model_RG.png', dpi=100, bbox_inches='tight', pad_inches=0)
plt.show()

print("RMSE:", rmse)
print("R²:", r2)

modelo_mqo = sm.OLS(y_train, X_train_mqo).fit()

#%% Fim!