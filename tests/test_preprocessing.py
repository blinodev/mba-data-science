import pandas as pd
from app.preprocessing import clean_data, engineer_features

def test_clean_data():
    df = pd.DataFrame({'feature': [1, None, 3]})
    cleaned = clean_data(df)
    assert cleaned.isnull().sum() == 0
    assert len(cleaned) == 2

def test_engineer_features():
    df = pd.DataFrame({'feature': [1, 2, 3]})
    engineered = engineer_features(df)
    assert 'feature_sq' in engineered.columns
    assert all(engineered['feature_sq'] == engineered['feature'] ** 2)
