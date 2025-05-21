import os
import pytest
from app.data import load_data, check_columns  # suas funções para carregar e validar dados

# Dicionário com os arquivos e colunas esperadas (ajuste conforme seus CSVs)
csv_files_info = {
    "labels.csv": ["period", "timedelta", "dst"],
    "solar_wind.csv": [
        "period", "timedelta", "bx_gse", "by_gse", "bz_gse", "theta_gse", "phi_gse",
        "bx_gsm", "by_gsm", "bz_gsm", "theta_gsm", "phi_gsm", "bt", "density",
        "speed", "temperature", "source"
    ],
    "sunspots.csv": ["period", "timedelta", "smoothed_ssn"],
}

@pytest.mark.parametrize("filename,expected_cols", csv_files_info.items())
def test_csv_files(filename, expected_cols):
    csv_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', filename))  # corrigido aqui
    df = load_data(csv_path)
    assert df is not None, f"Arquivo {filename} não carregou"
    assert not df.empty, f"Arquivo {filename} está vazio"
    ok, msg = check_columns(df, expected_cols)
    assert ok, f"{filename}: {msg}"
