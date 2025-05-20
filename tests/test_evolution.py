from app.evaluation import evaluate
import numpy as np

def test_evaluate():
    y_true = np.array([3, -0.5, 2, 7])
    y_pred = np.array([2.5, 0.0, 2, 8])
    mse = evaluate(y_true, y_pred)
    assert mse > 0
