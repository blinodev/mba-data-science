''' ← Logging e relatórios
'''

# src/logging.py

from datetime import datetime
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

def log_model_performance(model_name, y_true, y_pred, etapa='treino', log_file="logs/model_logs.log"):
    """
    Registra o desempenho do modelo em um arquivo de log
    
    Args:
        model_name: Nome do modelo
        y_true: Valores reais
        y_pred: Valores previstos
        etapa: 'treino' ou 'teste'
        log_file: Caminho do arquivo de log
    """
    metrics = {
        'R2': round(r2_score(y_true, y_pred), 4),
        'RMSE': round(mean_squared_error(y_true, y_pred), 4),
        
    }
    
    log_entry = f"[{datetime.now()}] {model_name} ({etapa}) - {metrics}\n"
    
    with open(log_file, 'a') as f:
        f.write(log_entry)
    
    return metrics

