import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.regression.linear_model import OLS

class LinearRegressionLeastSquare:
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("inconsistency between X and Y sample number!")
        elif self.X.shape[0] == 0:
            raise ValueError("input sample can not be empty!")
        elif self.Y.ndim != 1:
            raise ValueError("Y must be one dimensional")
        self.__predict__()

    def getSlope(self):
        return self.slope

    def getIntersect(self):
        return self.intersection

    def __predict__(self):
        meanX = np.mean(self.X, axis=0)
        meanY = np.mean(self.Y)
        self.slope = np.sum((self.X - meanX) * (self.Y - meanY)) / np.sum((self.X - meanX) ** 2)
        self.intersection = meanY - self.slope * meanX
        #print(self.slope, self.intersection)

class LinearRegressionGradientDescent:
    def __init__(self):
        self.w = None
        self.b = 0
        self.r2 = None

    def __predict__(self, X):
        return np.dot(X, self.w) + self.b

    def __error__(self, Y, y_pred):
        return 0.5 * np.mean(np.square((Y - y_pred)))

    def fit(self, X, Y, lr, max_it):
        X = np.asarray(X).astype(float)
        Y = np.asarray(Y).astype(float)

        if X.ndim == 1:
            X = X.reshape((Y.shape[0], -1))
        self.w = np.random.randn(X.shape[1])
        train_err = []
        for it in range(max_it):
            pred_y = self.__predict__(X)
            err = self.__error__(Y, pred_y)
            train_err.append(err)

            dw = np.dot(X.T, pred_y - Y) / X.shape[0]
            db = np.mean(pred_y - Y)
            self.w -= lr * dw
            self.b -= lr * db
        pred_y = self.__predict__(X)
        mean_y = np.mean(Y)
        self.r2 = 1 - np.sum(np.square(pred_y - Y)) / np.sum(np.square(mean_y - Y))
        print('W = {}, b = {}'.format(self.w, self.b))
        assert np.abs(self.r2 - r2_score(Y, pred_y)) <= 1e-13

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
        lr = LinearRegression()
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

if __name__ == "__main__":
    W = np.array([-3, 5, 1, 2, 4])
    intersect = -30
    X = np.random.randn(100, W.shape[0])
    Y = np.dot(X, W) + intersect + np.random.randn(X.shape[0]) * 0.05
    collinearityCheck(X)
    regression = LinearRegressionGradientDescent()
    regression.fit(X, Y, 0.01, 5000)
    y_pred = regression.__predict__(X)

    homoscedasticityCheck(Y, y_pred)
    normalDistributionCheck(Y, y_pred)

    # regSklearn = LinearRegression()
    # regSklearn.fit(X, Y)
    # print(regSklearn.coef_, regSklearn.intercept_)
