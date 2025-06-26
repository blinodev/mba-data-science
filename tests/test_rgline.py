import pytest
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from app.model_rgline import train_model, predict

@pytest.fixture
def sample_data():
    # Dados fictícios para treinamento e teste
    X = pd.DataFrame({
        'feature1': [1, 2, 3, 4],
        'feature2': [10, 20, 30, 40]
    })
    y = [100, 200, 300, 400]
    return X, y

def test_train_model(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    assert model is not None
    assert hasattr(model, "predict")

def test_predict_output_shape(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    preds = predict(model, X)
    assert len(preds) == len(X)

def test_model_metrics(sample_data):
    X, y = sample_data
    model = train_model(X, y)
    preds = predict(model, X)

    r2 = r2_score(y, preds)
    mae = mean_absolute_error(y, preds)
    mse = mean_squared_error(y, preds)
    rmse = np.sqrt(mse)

    # Testa se as métricas estão dentro dos limites aceitáveis para esse dataset perfeito
    assert r2 > 0.99
    assert mae < 1e-6
    assert mse < 1e-6
    assert rmse < 1e-3

    # print(f"R²: {r2:.1%} | MAE: {mae:.4f} | MSE: {mse:.6f} | RMSE: {rmse:.6f}")
