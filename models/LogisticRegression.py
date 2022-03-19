import numpy as np
from matplotlib import pyplot as plt


class LogisticRegression:
    def __init__(self, X, Y, lr, max_it = 100):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.lr = lr
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if X.shape[0] != Y.shape[0]:
            raise ValueError("Different sample number of X and Y")
        if X.shape[0] == 0:
            raise ValueError("Empty input samples")
        self.W = np.zeros(X.shape[1], dtype=float)
        self.costs = []

        cost = np.inf
        it = 0
        while (self.costs.__len__() == 0 or self.costs[-1] > 1e-5) and it < max_it:
            it += 1
            self.W += -lr * self.__gradient__(X, Y)
            cost = self.__cost__(X, Y)
            self.costs.append(cost)
        print(it)
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        if X.shape[1] != self.W.shape[0]:
            raise ValueError("input sample size error")
        return self.__sigmoid__(X)


    def __sigmoid__(self, X):
        return 1 / (1 + np.exp(-np.dot(X, self.W)))

    def __gradient__(self, X, Y):
        pred = self.__sigmoid__(X)
        return np.dot(X.T, pred - Y) / Y.shape[0]

    def __cost__(self, X, Y):
        pred = self.__sigmoid__(X)
        return np.mean(-Y * np.log(pred + 1e-11) - (1 - Y) * np.log(1 - pred + 1e-11))

if __name__ == "__main__":
    x1 = np.random.rand(5, 1) + 1
    x2 = np.random.rand(5, 1) - 3
    X = np.concatenate([x1, x2], axis=0)
    Y = np.concatenate([np.ones(5), -np.zeros(5)], axis=0)
    regression = LogisticRegression(X, Y, 1, 100000)
    plt.scatter(X, Y)

    xx = np.linspace(np.min(X), np.max(X), 100).reshape((-1, 1))
    yy = regression.predict(xx)
    plt.plot(xx, yy)
    plt.show()
