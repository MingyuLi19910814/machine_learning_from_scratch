import numpy as np
from Evaluate import evaluate_regressor
from sklearn import linear_model

class LinearRegression:
    def __init__(self, lr=0.1, max_it=500):
        self.lr = lr
        self.max_it = max_it
        self.w = None
        self.b = 0

    def __predict__(self, X):
        return np.dot(X, self.w) + self.b

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError("invalid input")
        self.w = np.random.randn(X.shape[1])
        for it in range(self.max_it):
            pred = self.__predict__(X)
            dw = np.dot(X.T, pred - Y) / Y.shape[0]
            db = np.mean(pred - Y)
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError("invalid input")
        return self.__predict__(X)

if __name__ == "__main__":
    reg1 = LinearRegression()
    reg2 = linear_model.LinearRegression()
    evaluate_regressor(reg1, reg2)