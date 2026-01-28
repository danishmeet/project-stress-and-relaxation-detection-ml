import numpy as np
from sklearn.preprocessing import StandardScaler

def normalize_features(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)
