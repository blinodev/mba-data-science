import numpy as np
from sklearn.preprocessing import StandardScaler

def padroniza_dados(x_treino, x_valid, x_teste):
    sc = StandardScaler()
    return sc.fit_transform(x_treino), sc.transform(x_valid), sc.transform(x_teste)

def dsa_ajusta_formato_dados(X_s, y_s, lag):
    X_train = []
    for variable in range(X_s.shape[1]):
        X = [X_s[i - lag:i, variable] for i in range(lag, X_s.shape[0])]
        X_train.append(X)
    X_train = np.swapaxes(np.swapaxes(np.array(X_train), 0, 1), 1, 2)
    y_train = [y_s[i].reshape(1, -1) for i in range(lag, y_s.shape[0])]
    return X_train, np.concatenate(y_train, axis=0)
