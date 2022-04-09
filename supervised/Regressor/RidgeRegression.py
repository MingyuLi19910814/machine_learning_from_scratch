import numpy as np
from sklearn.linear_model import Ridge
from Evaluate import evaluate_regressor

class RidgeRegression:
    def __init__(self, lr=0.1, max_it=1500, alpha=1):
        self.w = None
        self.b = 0
        self.alpha = alpha
        self.lr = lr
        self.max_it = max_it

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
            dw = np.dot(X.T, pred - Y) / Y.shape[0] + 1 * self.alpha * self.w / Y.shape[0]
            db = np.mean(pred - Y) + 1 * self.alpha * self.b
            self.w -= self.lr * dw
            self.b -= self.lr * db

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError("invalid input")
        return self.__predict__(X)

if __name__ == "__main__":
    reg1 = RidgeRegression()
    reg2 = Ridge(alpha=1, max_iter=1500)
    evaluate_regressor(reg1, reg2, n_samples=10000)