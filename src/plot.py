
"""
Visualização de resultados
"""

# src/plot.py

import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def estilizar_spines(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

def plotar_comparacao_modelos(nomes_modelos, r2_scores, rmse_scores, nome_modelo='modelo', salvar_em=None):
    """
    Gera gráficos comparativos de R² e RMSE entre os modelos, ordenando do melhor para o pior.

    Parâmetros:
    -----------
    nomes_modelos : list
        Lista com os nomes dos modelos.

    r2_scores : list
        Lista com os valores de R².

    rmse_scores : list
        Lista com os valores de RMSE.

    nome_modelo : str, opcional
        Nome base do modelo para uso no nome do arquivo final.

    salvar_em : str, opcional
        Caminho completo para salvar o gráfico final. Se None, será gerado com base no nome_modelo.
    """
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns

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
    plt.show()

    

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
    plt.show()

    print(f"📊 Gráfico salvo em: {salvar_em}")
    
    
def plotar_distribuicao_treino_teste(X_train, X_test, nome_modelo='modelo', salvar_em=None):
    """
    Plota gráfico de barras horizontais mostrando a distribuição treino/teste.

    Parâmetros:
    -----------
    X_train : array-like
        Dados de treino
    X_test : array-like  
        Dados de teste
    nome_modelo : str, opcional
        Nome do modelo para uso no nome do arquivo
    salvar_em : str, opcional
        Caminho completo para salvar o gráfico. Se None, será salvo em 'output/desemp_model_<nome_modelo>.png'
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import os

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

    # Aplicando estilo às spines (se função existir)
    try:
        estilizar_spines(ax)
    except NameError:
        pass  # Ignora se a função não estiver definida

    ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
    ax.set_xlim(0, max(n_train, n_test) * 1.1)
    ax.legend(frameon=False, bbox_to_anchor=(1, 1), loc='upper left')

    plt.tight_layout()

    # Caminho de salvamento
    if salvar_em is None:
        salvar_em = f"output/desemp_model_{nome_modelo.replace(' ', '_').lower()}.png"

    os.makedirs(os.path.dirname(salvar_em), exist_ok=True)
    plt.savefig(salvar_em, dpi=100, bbox_inches='tight', pad_inches=0)
    plt.show()

    print(f"📊 Gráfico salvo em: {salvar_em}")

    
  