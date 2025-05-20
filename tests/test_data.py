import pytest
from app.data import load_data, check_columns

def test_load_data():
    df = load_data('tests/sample_data.csv')
    assert not df.empty

def test_check_columns():
    df = load_data('tests/sample_data.csv')
    assert check_columns(df, ['id', 'feature', 'target'])
