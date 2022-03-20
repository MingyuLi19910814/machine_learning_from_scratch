import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    def __init__(self, X, Y, lr, max_it):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.shape[0] != Y.shape[0] or X.shape[0] == 0:
            raise ValueError('input dimension error')
        if max_it <= 0:
            raise ValueError('iteration must be positive')
        if lr <= 0:
            raise ValueError('learning rate must be positive')
        self.W = np.zeros(X.shape[1])
        self.costs = []
        lastCost = np.inf
        it = 0
        while it < max_it:
            it += 1
            cost = self.__cost__(X, Y)
            self.W += -lr * self.__gradient__(X, Y)
            self.costs.append(cost)
            if abs(cost - lastCost) < 1e-5:
                break
            lastCost = cost

    def predict(self, X):
        X = np.asarray(X, dtype=float)
        X = X.reshape((-1, self.W.shape[0]))
        return self.__sigmoid__(X)

    def __sigmoid__(self, X):
        return 1 / ( 1 + np.exp(-np.dot(X, self.W)))

    def __gradient__(self, X, Y):
        pred = self.__sigmoid__(X)
        return np.dot(X.T, pred - Y) / Y.shape[0]

    def __cost__(self, X, Y):
        pred = self.__sigmoid__(X)
        return np.mean(-Y * np.log(pred) - (1 - Y) * np.log(1 - pred))

if __name__ == "__main__":
    x1 = np.random.rand(5, 1) + 1
    x2 = np.random.rand(5, 1) - 3
    X = np.concatenate([x1, x2], axis=0)
    Y = np.concatenate([np.ones(5), -np.zeros(5)], axis=0)
    regression = LogisticRegression(X, Y, 0.1, 1000)
    plt.scatter(X, Y)

    xx = np.linspace(np.min(X), np.max(X), 100).reshape((-1, 1))
    yy = regression.predict(xx)
    plt.plot(xx, yy)
    plt.show()
