from sklearn import linear_model
import numpy as np
from Evaluate import evaluate_classifier

class LogisticRegression:
    def __init__(self, lr, max_it):
        self.w = None
        self.b = 0
        self.lr = lr
        self.max_it = max_it

    def __predict__(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.w) - self.b))

    def __loss__(self, X, gt):
        pred = self.__predict__(X)
        epsilon = 1e-13
        return -np.sum(gt * np.log(pred + epsilon) + (1 - gt) * np.log(1 - pred + epsilon))

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
            self.w -= dw * self.lr
            self.b -= db * self.lr

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError('invalid iput')
        Y = self.__predict__(X)
        return Y >= 0.5


if __name__ == "__main__":
    cls1 = LogisticRegression(lr=0.01, max_it=5000)
    cls2 = linear_model.LogisticRegression()
    evaluate_classifier(cls1, cls2)