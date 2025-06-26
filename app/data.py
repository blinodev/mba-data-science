import pandas as pd

def load_data(path):
    """Carrega um CSV e retorna um DataFrame."""
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        print(f"❌ Arquivo não encontrado: {path}")
        return None

def save_data(df, path):
    """Salva um DataFrame como CSV."""
    df.to_csv(path, index=False)
    print(f"✅ Dados salvos em {path}")

def check_columns(df, expected_columns=None):
    """
    Verifica se todas as colunas esperadas estão presentes no DataFrame.
    """
    if expected_columns is None:
        expected_columns = [
            "period", "timedelta", "bx_gse", "by_gse", "bz_gse", "theta_gse", "phi_gse",
            "bx_gsm", "by_gsm", "bz_gsm", "theta_gsm", "phi_gsm", "bt", "density",
            "speed", "temperature", "source"
        ]
    missing = [col for col in expected_columns if col not in df.columns]
    if missing:
        return False, f"Colunas ausentes: {missing}"
    return True, "Todas as colunas presentes"
