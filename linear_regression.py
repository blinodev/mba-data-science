# -*- coding: utf-8 -*-
"""
Created on Tue Feb 25 09:27:43 2025

@author: familia
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


#%% Importar a base (já tratada)
dados = pd.read_pickle('df_final.pkl')

#%%
def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

relatorio_missing(dados)

# In[5.5]: Modelo de Regressão Linear Múltipla (MQO)

# Estimação do modelo
reg = sm.OLS.from_formula(formula = 'dst ~ vel +\
                                     theta_gsm + temp + mag +\
                                         bz_gsm + med', data=dados).fit()
# Obtenção dos outputs
reg.summary()

#%% Teste de verificação da aderência dos resíduos à normalidade

# Elaboração do teste de Shapiro-Francia
teste_sf = shapiro_francia(reg.resid)
round(teste_sf['p-value'], 5)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_sf['p-value'] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#%% Histograma dos resíduos do modelo OLS

# Parâmetros de referência para a distribuição normal teórica
(mu, std) = norm.fit(reg.resid)

# Criação do gráfico
plt.figure(figsize=(15,10))
plt.hist(reg.resid, bins=35, density=True, alpha=0.7, color='Blue')
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 1000)
p = norm.pdf(x, mu, std)
plt.plot(x, p, linewidth=3, color='red')
plt.title('Resíduos do Modelo', fontsize=20)
plt.xlabel('Resíduos', fontsize=22)
plt.ylabel('Frequência', fontsize=22)
plt.show()

#%% Realizando a transformação de Box-Cox na variável dependente
from scipy.stats import yeojohnson

# Realizando a transformação de Yeo-Johnson
y_box, lmbda = yeojohnson(dados['dst'])

# Valor obtido para o lambda
print(lmbda)

# Adicionando ao banco de dados
dados['dst_bc'] = y_box

# In[5.5]: Modelo de regressão com transformação de yeojohnson em Y

# Estimação do modelo
reg_bc = sm.OLS.from_formula(formula = 'dst_bc ~ vel +\
                                     theta_gsm + temp + mag +\
                                         bz_gsm + med', data=dados).fit()
# Obtenção dos outputs
reg_bc.summary()

#%% Reavaliando aderência à normalidade dos resíduos do modelo

# Teste de Shapiro-Francia
teste_sf_bc = shapiro_francia(reg_bc.resid)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_sf_bc['p-value'] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

#%% Removendo as variáveis que não apresentam significância estatística

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# Stepwise do modelo
modelo_stepwise_bc = stepwise(reg_bc, pvalue_limit=0.05)

# Teste de Shapiro-Francia
teste_sf_step = shapiro_francia(modelo_stepwise_bc.resid)

#%% Novo histograma dos resíduos do modelo

# Parâmetros de referência para a distribuição normal teórica
(mu_bc, std_bc) = norm.fit(modelo_stepwise_bc.resid)

# Criação do gráfico
plt.figure(figsize=(15,10))
plt.hist(modelo_stepwise_bc.resid, bins=30, density=True, alpha=0.8, color='Blue')
xmin_bc, xmax_bc = plt.xlim()
x_bc = np.linspace(xmin_bc, xmax_bc, 1000)
p_bc = norm.pdf(x_bc, mu_bc, std_bc)
plt.plot(x_bc, p_bc, linewidth=3, color='red')
plt.title('Resíduos do Modelo Box-Cox Yeo Johnson', fontsize=20)
plt.xlabel('Resíduos', fontsize=22)
plt.ylabel('Frequência', fontsize=22)
plt.show()

#%%
dados['fitted_bc'] = reg_bc.predict(dados)

print(reg_bc.summary())

dados.head()
print(lmbda)

print(dados[['dst', 'dst_bc', 'fitted_bc', 'fitted_original']].head())

print(dados['fitted_bc'].describe())  # Estatísticas básicas
print(dados['fitted_bc'].isnull().sum())  # Verifique valores nulos

#%% import numpy as np
import numpy as np


# Verifique se (lmbda * fitted_bc + 1) é positivo
valid_mask = (lmbda * dados['fitted_bc'] + 1) > 0
print("Número de valores inválidos:", (~valid_mask).sum())

# Aplique a retransformação apenas para valores válidos
dados.loc[valid_mask, 'fitted_original'] = (lmbda * dados.loc[valid_mask, 'fitted_bc'] + 1) ** (1 / lmbda) - 1

# Substitua valores inválidos por 0 (ou outro valor adequado)
dados['fitted_original'] = dados['fitted_original'].fillna(0)  # Corrigido aqui

# Exiba os resultados
print(dados[['dst', 'dst_bc', 'fitted_bc', 'fitted_original']].head())

print(dados['fitted_bc'].describe())

# Aplique a retransformação apenas para valores válidos
valid_mask = (lmbda * dados['fitted_bc'] + 1) > 0
dados.loc[valid_mask, 'fitted_original'] = (lmbda * dados.loc[valid_mask, 'fitted_bc'] + 1) ** (1 / lmbda) - 1

# Substitua valores inválidos pela média de fitted_original válida
mean_fitted_original = dados.loc[valid_mask, 'fitted_original'].mean()
dados.loc[~valid_mask, 'fitted_original'] = mean_fitted_original

#%% Realizando predições com base no modelo estimado

modelo_stepwise_bc.params
reg_bc.params

# Modelo Não Linear:
valor_pred_bc = modelo_stepwise_bc.predict(pd.DataFrame({
    'vel': [-7.227439],
    'theta_gsm': [2.461796],
    'temp':[  1.997853],
    'mag': [-0.794493],
    'bz_gsm': [ 0.251232],
    'med': [-0.877940]
}))

# Valor predito pelo modelo BC
print(f"Valor Predito: {round(valor_pred_bc[0], 2)}")

# Cálculo inverso para a obtenção do valor predito Y (preço)
valor_pred_dst = (valor_pred_bc * lmbda + 1) ** (1 / lmbda)
print(f"Valor Predito (Dst): {round(valor_pred_dst[0], 2)}")

#%% Gráfico fitted values

# Valores preditos pelo modelo para as observações da amostra
dados['fitted_bc'] = modelo_stepwise_bc.predict()

sns.regplot(dados, x='dst', y='fitted_bc', color='blue', ci=False, line_kws={'color': 'red'})
plt.title('Analisando o Ajuste das Previsões', fontsize=10)
plt.xlabel('Dst Observado', fontsize=10)
plt.ylabel('Dst Previsto', fontsize=10)
plt.axline((5.95, 5.95), (max(dados['dst']), max(dados['dst'])), linewidth=1, color='grey')
plt.show()


#%% Criação da função para o teste de Breusch-Pagan (heterocedasticidade)

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return chisq, p_value

#%% Aplicando a função criada para realizar o teste

teste_bp = breusch_pagan_test(modelo_stepwise_bc)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_bp[1] > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

#%% Analisando a presença de heterocedasticidade no modelo original

teste_bp_original = breusch_pagan_test(reg)

# Tomando a decisão por meio do teste de hipóteses

alpha = 0.05 # nível de significância do teste

if teste_bp_original[1] > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

## O modelo com a transformação de Box-Cox ajustou os termos
## de erros heterocedásticos, indicando potencial erro 
## da forma funcional do modelo originalmente estimado

#%% Cálculo das métricas
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

r2 = r2_score(dados['dst'], dados['fitted_bc'])
mae = mean_absolute_error(dados['dst'], dados['fitted_bc'])
mse = mean_squared_error(dados['dst'], dados['fitted_bc'])
rmse = np.sqrt(mse)

# Exibindo resultados
print("Resumo do Modelo:")
print(f"R²: {r2:.1%}")
print(f"MAE: {mae:.2f}")
print(f"MSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")

#%% Fim!