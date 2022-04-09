import numpy as np
from sklearn import ensemble
from sklearn.tree import DecisionTreeRegressor
from Evaluate import evaluate_regressor
from matplotlib import pyplot as plt


class GradientBoostingRegressor:
    def __init__(self, n_estimators, max_depth, lr):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.lr = lr
        self.estimators = [DecisionTreeRegressor(max_depth=max_depth) for _ in range(self.n_estimators)]
        self.initial_pred = None
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError("invalid input")
        self.initial_pred = np.mean(Y)
        pred = self.initial_pred
        for idx in range(self.n_estimators):
            residual = Y - pred
            self.estimators[idx].fit(X, residual)
            pred = pred + self.lr * self.estimators[idx].predict(X)

    def predict(self, X):
        X = np.asarray(X)
        pred = np.ones(X.shape[0]) * self.initial_pred
        for estimator in self.estimators:
            pred = pred + self.lr * estimator.predict(X)
        return pred

if __name__ == "__main__":
    n_estimators = 100
    lr = 0.1
    max_depth = 2
    reg1 = GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, lr=lr)
    reg2 = ensemble.GradientBoostingRegressor(n_estimators=n_estimators, max_depth=max_depth, learning_rate=lr)
    evaluate_regressor(reg1, reg2)