import numpy as np
from app.model import train_model, predict

def test_train_and_predict():
    X = np.array([[1], [2], [3]])
    y = np.array([2, 4, 6])
    model = train_model(X, y)
    preds = predict(model, X)
    assert len(preds) == len(y)
    assert all(preds > 0)
