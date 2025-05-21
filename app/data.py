import pandas as pd

def load_data(path):
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"Arquivo não encontrado: {path}")
        return None

def save_data(df, path):
    df.to_csv(path, index=False)
    print(f"Dados salvos em {path}")

def save_data(df, filepath):
    df.to_csv(filepath, index=False)
    print(f"Dados salvos em {filepath}")

