import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn import linear_model

class LinearRegressionUnivariate:
    def __init__(self):
        self.w = 0
        self.b = 0

    def fit(self, X, Y):
        mean_x = np.mean(X)
        mean_y = np.mean(Y)
        self.w = np.sum((X - mean_x) * (Y - mean_y)) / np.sum((X - mean_x) ** 2)
        self.b = mean_y - self.w * mean_x
        print('w = {}, b = {}'.format(self.w, self.b))

class LinearRegressionMultivariate:
    def __init__(self):
        self.w = None
        self.b = None

    def fit(self, X, Y, lr=0.1, max_it=500):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or Y.ndim != 1 or X.shape[0] != Y.shape[0] or X.shape[0] == 0:
            raise ValueError("invalid input")
        self.w = np.random.randn(X.shape[1])
        self.b = 0
        for it in range(max_it):
            pred = self.__predict__(X)
            loss = 0.5 * np.mean((pred - Y) ** 2)
            dw = np.dot(X.T, pred - Y) / Y.shape[0]
            db = np.mean(pred - Y)
            self.w -= lr * dw
            self.b -= lr * db
        print('w = {}, b = {}'.format(self.w, self.b))

    def __predict__(self, X):
        return np.dot(X, self.w) + self.b


def plot(X, Y, w, b):
    plt.scatter(X, Y)
    y_pred = np.dot(X, w) + b
    plt.plot(X, y_pred)
    plt.show()

def collinearityCheck(X):
    print('*' * 40)
    print('collinearity check!')
    def calc_vif(X, idx):
        y = X[:, idx]
        include_idx = np.arange(X.shape[1]) != idx
        x = X[:, include_idx]
        lr = linear_model.LinearRegression()
        lr.fit(x, y)
        y_pred = lr.predict(x)
        r2 = r2_score(y, y_pred)
        vif = 1 / (1 - r2)
        return vif
    for feature_idx in range(X.shape[1]):
        vif = calc_vif(X, feature_idx)
        print('idx = {}, vif = {}'.format(feature_idx, vif))


def homoscedasticityCheck(Y, y_pred):
    print('*' * 40)
    print('homoscedasticity check!')
    res = Y - y_pred
    plt.scatter(Y, res)
    plt.show()

def normalDistributionCheck(Y, y_pred):
    print('*' * 40)
    print('normal distribution check!')
    res = Y - y_pred
    plt.hist(res)
    plt.show()

def univariate_regression():
    w = 3.2
    b = 20
    x = np.random.randn(100)
    y = w * x + b
    lr = LinearRegressionUnivariate()
    lr.fit(x, y)

    lr = linear_model.LinearRegression()
    lr.fit(x.reshape((-1, 1)), y.reshape((-1, 1)))
    print('sklearn w = {}, b = {}'.format(lr.coef_, lr.intercept_))


def multivariate_regression():
    W = np.array([-3, 5, 1, 2, 4])
    intersect = -30
    X = np.random.randn(100, W.shape[0])
    Y = np.dot(X, W) + intersect + np.random.randn(X.shape[0]) * 0.05
    collinearityCheck(X)
    regression = LinearRegressionMultivariate()
    regression.fit(X, Y, 0.01, 5000)
    lr = linear_model.LinearRegression()
    lr.fit(X, Y)
    print('sklearn w = {}, b = {}'.format(lr.coef_, lr.intercept_))
    y_pred = regression.__predict__(X)

    homoscedasticityCheck(Y, y_pred)
    normalDistributionCheck(Y, y_pred)

if __name__ == "__main__":
    #univariate_regression()
    multivariate_regression()
