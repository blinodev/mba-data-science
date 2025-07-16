# utils.py
import pandas as pd
from IPython.display import display, Markdown

# Relatório de valores faltantes
def relatorio_missing(df):
    print(f'Número de linhas: {df.shape[0]} | Número de colunas: {df.shape[1]}')
    return pd.DataFrame({
        'Pct_missing': df.isna().mean().apply(lambda x: f"{x:.1%}"),
        'Freq_missing': df.isna().sum()
    })
#---------------------------------------------------------------------------------------------
# Resumo dos dados com exibição formatada
def resumo_dados(nome, df):
    """
    Exibe um resumo informativo sobre um DataFrame, incluindo:
    - Primeiras linhas
    - Tipos de dados, quantidade e percentual de valores nulos
    """
    display(Markdown(f"### 📌 {nome}"))
    
    display(Markdown("**🔹 Primeiras linhas:**"))
    display(df.head())

    display(Markdown("**🔹 Informações da Tabela:**"))
    df_info = df.dtypes.to_frame(name='Tipo')
    df_info['Nulos'] = df.isnull().sum()
    df_info['% Nulos'] = (df.isnull().mean() * 100).round(1).astype(str) + '%'
    display(df_info)

# Estatísticas descritivas por dataset
def visualizar_descritivas_dst(dst):
    return dst.groupby("per").describe()

def visualizar_descritivas_manchas(sunspots):
    return sunspots.groupby(['per'])['med'].describe().T

def visualizar_descritivas_vento_solar(solar_wind):
    return solar_wind.groupby("per").describe().T

#------------------------------------------------------------------
# Função para plotagem descritiva (pointplot + countplot)
def descritiva(df_, var, vresp='dst', max_classes=5):
    df = df_.copy()

    # Discretizar se houver muitas classes
    if df[var].nunique() > max_classes:
        df[var] = pd.qcut(df[var], max_classes, duplicates='drop')

    # Criar a figura e primeiro eixo
    fig, ax1 = plt.subplots(figsize=(10, 6))
    sns.pointplot(data=df, y=vresp, x=var, ax=ax1, color='black')

    # Segundo eixo compartilhado (frequência)
    ax2 = ax1.twinx()
    sns.countplot(data=df, x=var, hue=var, palette='viridis', alpha=0.5, ax=ax2, legend=False)

    # Estética do eixo de frequência
    ax2.set_ylabel('Frequência', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.grid(False)

    # Estética do eixo principal
    ax1.set_ylabel(vresp, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_zorder(2)
    ax1.patch.set_visible(False)
    ax1.grid(False)

    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------

# Função para plotar histograma do índice dst com hue
def plot_histograma_dst(df, coluna_valor='dst', coluna_hue='period', bins=20, salvar=False):
    plt.figure(figsize=(8, 6))
    sns.histplot(data=df, x=coluna_valor, hue=coluna_hue, bins=bins, palette='Set2', multiple='layer')

    plt.xlabel('Intensidade da Perturbação Geomagnética (nT)')
    plt.ylabel('Frequência')

    plt.xlim(-150, 50)  # Ajuste conforme necessidade
    plt.ylim(0, 35000)  # Ajuste conforme necessidade

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

    if salvar:
        plt.savefig('dst_distri.png', dpi=100, bbox_inches='tight', pad_inches=0.1)
    plt.show()

#----------------------------------------------------------------------
# Função para plotar boxplot de uma variável por período
def plot_boxplot_dst(df, coluna_x='per', coluna_y='dst', palette='Set2'):
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=coluna_x, y=coluna_y, hue=coluna_x, data=df, palette=palette, dodge=False)

    plt.xlabel(coluna_x.capitalize())
    plt.ylabel(coluna_y.capitalize())

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)

#--------------------------------------------------------------------
def plot_sunspots_boxplot(df):  
    plt.figure(figsize=(8, 6))
    sns.boxplot(
        x='period', 
        y='smoothed_ssn', 
        hue='period', 
        data=df,  
        palette="Set2",
        dodge=False
    )

# ----------------------------------------------------------------
