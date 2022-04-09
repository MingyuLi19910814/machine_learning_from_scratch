import numpy as np
from Evaluate import evaluate_regressor
from sklearn.neighbors import KNeighborsRegressor

class KNearestNeighborRegressor:
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.X.shape[1]:
            raise ValueError("invalid input")
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.X, axis=-1)
        idx = np.argsort(dists, axis=-1)[:, :self.k]
        pred = np.asarray([np.mean(self.Y[i]) for i in idx])
        return pred

if __name__ == "__main__":
    n_neighbors = 5
    reg1 = KNearestNeighborRegressor(n_neighbors)
    reg2 = KNeighborsRegressor(n_neighbors=n_neighbors)
    evaluate_regressor(reg1, reg2)