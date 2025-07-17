
"""
Created on Tue Jul 15 10:49:20 2025

@author: familia
"""

from pathlib import Path
import pandas as pd

def carregar_dados(caminho):
    """
    Carrega um arquivo pickle e retorna um DataFrame.
    """
    try:
        caminho = Path(caminho)
        if not caminho.exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {caminho}")
        
        dados = pd.read_pickle(caminho)

        if not isinstance(dados, pd.DataFrame):
            raise ValueError("O arquivo pickle não contém um DataFrame")

        print(f"✅ Dados carregados com sucesso! Shape: {dados.shape}")
        return dados

    except Exception as e:
        print(f"❌ Erro ao carregar {caminho}: {str(e)}")
        raise

from sklearn.model_selection import train_test_split

def separar_dados_treino_teste(dados, target='dst', test_size=0.3, random_state=100, plot=False):
    """
    Separa os dados em conjuntos de treino e teste, 
    com opção de visualizar a divisão.

    Retorna:
    --------
    tuple: (X_train, X_test, y_train, y_test)
    """
    X = dados.drop(columns=[target])
    y = dados[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if plot:
        plotar_distribuicao_treino_teste(X_train, X_test)

    return X_train, X_test, y_train, y_test



def preparar_dados_transformer(dados, coluna_alvo='dst', test_size=0.3, random_state=42, salvar_plot=True):
    """
    Separa os dados em treino e teste. Pode salvar gráfico da divisão.
    """
    if coluna_alvo not in dados.columns:
        raise ValueError(f"Coluna alvo '{coluna_alvo}' não encontrada nos dados.")

    X = dados.drop(columns=[coluna_alvo])
    y = dados[coluna_alvo]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if salvar_plot:
        plotar_distribuicao_treino_teste(X_train, X_test)

    return X_train, X_test, y_train, y_test


def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
                          'Freq_missing': df.isna().sum().apply(lambda x: f"{x:,.0f}").replace(',','.')})

import matplotlib.pyplot as plt
import numpy as np
import os

def plotar_distribuicao_treino_teste(X_train, X_test, nome_modelo='modelo', salvar_em=None):
    # Função interna para estilizar spines
    def estilizar_spines(ax):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

    # Configurações do gráfico
    ind = [0, 1]  # Posições das barras
    nomes = ["Treinamento", "Teste"]
    largura = 0.75
    cmap = plt.get_cmap("viridis")
    cores = cmap(np.linspace(0.2, 0.8, 2))  # Cores suaves

    # Cálculo das quantidades e percentuais
    n_train = len(X_train)
    n_test = len(X_test)
    total = n_train + n_test
    percent_train = (n_train / total) * 100
    percent_test = (n_test / total) * 100

    # Criando a figura
    fig, ax = plt.subplots(figsize=(8, 4))  # Altura reduzida para melhor proporção

    # Plotando as barras
    ax.barh(ind[0], n_train, largura, 
            label=f"Treinamento ({percent_train:.1f}%)", 
            color=cores[0])
    ax.barh(ind[1], n_test, largura, 
            label=f"Teste ({percent_test:.1f}%)", 
            color=cores[1])

    # Adicionando rótulos dentro das barras
    ax.text(n_train/2, ind[0], f"{n_train:,} ({percent_train:.1f}%)".replace(",", "."), 
            va='center', ha='center', color="white", 
            fontsize=11, fontweight='bold')
    ax.text(n_test/2, ind[1], f"{n_test:,} ({percent_test:.1f}%)".replace(",", "."), 
            va='center', ha='center', color="black", 
            fontsize=11, fontweight='bold')

    # Formatando o gráfico
    ax.set_yticks(ind)
    ax.set_yticklabels(nomes, fontsize=11)
    ax.set_xlabel("Número de Amostras", fontsize=11)

    # Aplicando estilo às spines
    estilizar_spines(ax)

    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlim(0, max(n_train, n_test) * 1.1)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

    plt.tight_layout()
    
    
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import seaborn as sns

def estilizar_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
  
def exibir_model(modelo, X_test, y_test, nome_modelo="Modelo", salvar_em=None):
    """
    Exibe as métricas e plota gráfico de desempenho do modelo.
    """
    y_pred = modelo.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"\n📌 Resumo do Modelo - {nome_modelo}")
    print(f"R²: {r2:.1%}")
    print(f"MAE: {mae:.2f}")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")

    metrics = ['R²', 'MSE', 'RMSE', 'MAE']
    values = [r2, mse, rmse, mae]
    df_metrics = pd.DataFrame({'Métrica': metrics, 'Valor': values})
    df_metrics_sorted = df_metrics.sort_values('Valor', ascending=False)

    plt.figure(figsize=(8, 6))
    # ax = sns.barplot(y='Métrica', x='Valor', data=df_metrics_sorted, palette='viridis')
    ax = sns.barplot(y='Métrica', x='Valor', hue='Métrica', data=df_metrics_sorted, palette='viridis', legend=False)

    for p in ax.patches:
        plt.annotate(
            f"{p.get_width():.2f}",
            xy=(p.get_width() * 1.005, p.get_y() + p.get_height() / 2),
            va='center',
            fontsize=10,
            fontweight='bold'
        )

    estilizar_spines(ax)
    plt.tight_layout()

    if salvar_em is None:
        salvar_em = f"output/desemp_model_{nome_modelo.replace(' ', '_').lower()}.png"

    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)

    plt.savefig(salvar_em, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"📊 Gráfico salvo em: {salvar_em}")
    
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def plotar_arvore_decisao(modelo, X, titulo="Árvore de Decisão", salvar_em=None):
    """
    Plota a árvore de decisão treinada.

    Args:
        modelo: Modelo DecisionTreeRegressor treinado
        X: DataFrame de treino (usado apenas para pegar os nomes das features)
        titulo: Título do gráfico
        salvar_em: Caminho para salvar a figura (opcional)
    """
    plt.figure(figsize=(20, 10), dpi=600)
    plot_tree(
        modelo,
        feature_names=X.columns.tolist(),
        filled=True,
        node_ids=True,
        precision=2
    )
    plt.title(titulo, fontsize=16)

    if salvar_em:
        import os
        os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
        plt.savefig(salvar_em, bbox_inches='tight')
        plt.close()

    


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def exibir_importancia_variaveis(modelo, X_train, salvar_em="output/importancia_variaveis.png"):
    """
    Exibe a importância das variáveis de um modelo de árvore e salva o gráfico.

    Parâmetros:
        modelo: Modelo treinado (por exemplo, DecisionTreeRegressor)
        X_train: DataFrame com as variáveis preditoras de treino
        salvar_em: Caminho para salvar o gráfico
    """
    # Criar DataFrame com importância
    tree_features = pd.DataFrame({
        'features': X_train.columns.tolist(),
        'importance': modelo.feature_importances_
    }).sort_values(by='importance', ascending=False).reset_index(drop=True)

    print(tree_features)

    # Criar gráfico
    plt.figure(figsize=(8, 6))
    ax = sns.barplot(
    x='importance', 
    y='features', 
    hue='features',  # Opcional: só é necessário se quiser cores diferentes por feature
    data=tree_features, 
    palette='viridis', 
    legend=False
    )
    plt.xlabel('Importância da Variável')
    plt.ylabel('Variáveis Preditoras')

    # Adicionar valores nas barras
    for index, value in enumerate(tree_features['importance']):
        plt.text(value, index, f'{value:.1%}', va='center', color='black', fontweight='bold', fontsize=12)

    # Estilização
    plt.tight_layout()
    estilizar_spines(ax)

    # Garantir pasta de saída
    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    plt.savefig(salvar_em, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    

    print(f"📊 Gráfico salvo em: {salvar_em}")
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plotar_fitted_values(y_true, y_pred, nome_modelo="Modelo", salvar_em=None, y_label="Previsto"):
    """
    Plota os valores reais vs preditos de um modelo (scatterplot).
    """
    graph = pd.DataFrame({'Real': y_true, 'Previsto': y_pred})

    plt.figure(dpi=600)
    ax = sns.scatterplot(graph, x='Real', y='Previsto', color='blue')
    # plt.title(f'Analisando as Previsões - {nome_modelo}', fontsize=10)
    plt.xlabel('Valor Real', fontsize=10)
    plt.ylabel(y_label, fontsize=10)

    max_val = max(graph.max())
    plt.axline((0, 0), (max_val, max_val), linewidth=1, color='grey', linestyle='--')

    plt.tight_layout()
    estilizar_spines(ax)
    
    if salvar_em:
        os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
        plt.savefig(salvar_em, dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"📊 Gráfico salvo em: {salvar_em}")
    
    


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from collections import OrderedDict


def prever_transformer(modelo, X):
    """
    Função para gerar previsões para modelos transformer.
    Aqui deve estar a lógica para fazer previsões com seu transformer.
    Substitua pelo código real de inferência do transformer.
    """
    # Exemplo dummy - substituir pela inferência real
    return modelo.predict(X)


def avaliar_modelo(modelo, X, y, tipo_modelo='sklearn'):
    """
    Avalia um modelo de regressão e retorna métricas de desempenho ordenadas por valor decrescente.

    Args:
        modelo: modelo treinado (deve ter o método .predict)
        X: dados preditores
        y: valores reais
        tipo_modelo: tipo do modelo ('sklearn', 'transformer')

    Returns:
        OrderedDict com R2, MSE, RMSE e MAE (maior valor primeiro)
    """
    if tipo_modelo not in ['sklearn', 'transformer']:
        raise ValueError(f"Tipo de modelo desconhecido: {tipo_modelo}. Use 'sklearn' ou 'transformer'.")

    try:
        if tipo_modelo == 'transformer':
            y_pred = prever_transformer(modelo, X)
        else:
            y_pred = modelo.predict(X)
    except Exception as e:
        print(f"❌ Erro ao prever: {e}")
        return OrderedDict()  # retorna dict vazio

    metricas = {
        'R2': r2_score(y, y_pred),
        'MSE': mean_squared_error(y, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y, y_pred)),
        'MAE': mean_absolute_error(y, y_pred)
    }

    # R2 primeiro, os demais em ordem decrescente
    metricas_ordenadas = OrderedDict(
        sorted(metricas.items(), key=lambda x: (x[0] != 'R2', -x[1] if x[0] != 'R2' else -x[1]))
    )

    return metricas_ordenadas


def avaliar_transformer(modelo, X_test, y_test):
    """
    Avalia o modelo transformer e imprime as métricas principais.
    """
    y_pred = prever_transformer(modelo, X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"R²: {r2:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")

    return {"R2": r2, "MAE": mae, "MSE": mse, "RMSE": rmse}

def plotar_comparacao_modelos(nomes_modelos, r2_scores, rmse_scores, nome_modelo='modelo', salvar_em=None):
  
    if not (len(nomes_modelos) == len(r2_scores) == len(rmse_scores)):
        raise ValueError("Todos os vetores (nomes_modelos, r2_scores, rmse_scores) devem ter o mesmo comprimento.")

    df = pd.DataFrame({
        "Modelo": nomes_modelos,
        "R²": r2_scores,
        "RMSE": rmse_scores
    })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # ---------- Gráfico R² ----------
    df_r2 = df.sort_values("R²", ascending=False)
    ax1 = sns.barplot(x="R²", y="Modelo", hue="Modelo", data=df_r2, palette="viridis", dodge=False, legend=False, ax=axes[0])

    for i, v in enumerate(df_r2["R²"]):
        ax1.text(v + 0.01, i, f"{v:.2f}", va='center', fontsize=10, fontweight='bold')

    try:
        estilizar_spines(ax1)
    except NameError:
        pass

    ax1.set_title("Comparação de R²")
    ax1.set_xlabel("R²")
    ax1.set_ylabel("Modelo")

    # ---------- Gráfico RMSE ----------
    df_rmse = df.sort_values("RMSE", ascending=True)
    ax2 = sns.barplot(x="RMSE", y="Modelo", hue="Modelo", data=df_rmse, palette="magma", dodge=False, legend=False, ax=axes[1])

    for i, v in enumerate(df_rmse["RMSE"]):
        ax2.text(v + 0.5, i, f"{v:.2f}", va='center', fontsize=10, fontweight='bold')

    try:
        estilizar_spines(ax2)
    except NameError:
        pass

    ax2.set_title("Comparação de RMSE")
    ax2.set_xlabel("RMSE")
    ax2.set_ylabel("")

    plt.tight_layout()

    # ---------- Caminho de salvamento ----------
    if salvar_em is None:
        salvar_em = f"output/desemp_model_{nome_modelo.replace(' ', '_').lower()}.png"

    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    plt.savefig(salvar_em, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.close()
    


import matplotlib.pyplot as plt
import pandas as pd
import os

def plotar_resultados_deep_learning(modelo, X_test, y_test, prefixo_nome="deep", salvar_plot=True):
    """
    Plota os gráficos de:
    - Real vs Previsto (primeiras 100 amostras)
    - Histórico de métricas (loss, val_loss, etc)
    - Curva de perda separadamente

    Args:
        modelo: modelo treinado com atributo `.history.history`
        X_test (pd.DataFrame ou np.ndarray): Dados de teste (features)
        y_test (pd.Series ou np.ndarray): Valores reais do target
        prefixo_nome (str): Prefixo para os nomes dos arquivos salvos
        salvar_plot (bool): Se True, salva as imagens em arquivos; se False, exibe os gráficos
    """

    import os
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs("output", exist_ok=True)

    # === Previsões ===
    y_pred = modelo.predict(X_test).flatten()
    df_resultados = pd.DataFrame({
        'Real': y_test.values,
        'Previsto': y_pred
    })

    # === Gráfico Real vs Previsto ===
    plt.figure(figsize=(12, 5))
    plt.plot(df_resultados['Real'][:100], label='Real')
    plt.plot(df_resultados['Previsto'][:100], label='Previsto', linestyle='--')
    plt.title(f' Real vs Previsto - {prefixo_nome}')
    plt.xlabel('Amostras')
    plt.ylabel('Target')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"output/{prefixo_nome}_fitted_values.png")
    plt.close()
        
    

    # === Histórico de treinamento ===
    history_df = pd.DataFrame(modelo.history.history).select_dtypes(include=['number'])

    if not history_df.empty:
        # Todas as métricas
        history_df.plot(figsize=(10, 6))
        plt.title(f' Histórico de Métricas - {prefixo_nome}')
        plt.xlabel('Épocas')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"output/{prefixo_nome}_history_all_metrics.png")
        plt.close()
            
       

        # Apenas perda (loss/val_loss)
        if 'loss' in history_df.columns and 'val_loss' in history_df.columns:
            history_df[['loss', 'val_loss']].plot(figsize=(10, 6))
            plt.title(f' Curva de Perda - {prefixo_nome}')
            plt.xlabel('Épocas')
            plt.ylabel('Perda')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(f"output/{prefixo_nome}_loss_curve.png")
            plt.close()
            
            
            
