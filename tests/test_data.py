import sys
import os
import pytest
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.data import load_data, check_columns

# Caminho relativo para os arquivos
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Informações esperadas por dataset
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
def test_csv_structure(filename, expected_cols):
    path = os.path.join(BASE_DIR, filename)
    df = load_data(path)
    assert df is not None, f"{filename} não foi carregado"
    assert not df.empty, f"{filename} está vazio"
    ok, msg = check_columns(df, expected_cols)
    assert ok, f"{filename}: {msg}"
