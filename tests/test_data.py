from app.data import load_data, check_columns  # agora deve funcionar
import sys
import os
sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))


def test_load_data_com_arquivo_valido():
    df = load_data("app/labels.csv")
    assert df is not None
    assert not df.empty


def test_check_columns_com_colunas_corretas():
    df = load_data("app/labels.csv")
    expected_cols = ["col1", "col2", "col3"]  # ajuste conforme seu CSV real
    ok, msg = check_columns(df, expected_cols)
    assert ok, msg
