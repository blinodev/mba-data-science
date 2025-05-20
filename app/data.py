import pandas as pd


def load_data(path):
    try:
        df = pd.read_csv(path)
        return df
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Arquivo vazio: {path}")
        return None
    except Exception as e:
        print(f"Erro ao ler o arquivo {path}: {e}")
        return None


def check_columns(df, expected_cols):
    if df is None:
        return False, "DataFrame não foi carregado"

    df_cols = set(df.columns)
    expected_cols_set = set(expected_cols)

    if expected_cols_set.issubset(df_cols):
        return True, "Todas as colunas esperadas estão presentes"
    else:
        missing = expected_cols_set - df_cols
        return False, f"Faltando colunas: {missing}"
