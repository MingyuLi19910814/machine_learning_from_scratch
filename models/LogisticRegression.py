import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import linear_model
class LogisticRegression:
    def __init__(self):
        self.w = None
        self.b = None

    def __predict__(self, X):
        v = np.dot(X, self.w) + self.b
        return 1 / (1 + np.exp(-v))

    def fit(self, X, Y, lr=0.1, max_it=1000):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0]:
            raise ValueError("invalid input")
        self.w = np.random.randn(X.shape[1])
        self.b = 0
        for it in range(max_it):
            pred = self.__predict__(X)
            dw = np.dot(X.T, pred - Y) / Y.shape[0]
            db = np.mean(pred - Y)
            self.w -= lr * dw
            self.b -= lr * db

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.w.shape[0]:
            raise ValueError("invalid input")
        pred = self.__predict__(X)
        return pred >= 0.5

if __name__ == "__main__":
    n_samples = 1000
    n_classes = 2
    n_features = 5
    n_informative = 2
    X, Y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=0)
    X = StandardScaler().fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    lr = LogisticRegression()
    lr.fit(train_x, train_y)
    pred_y = lr.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('acc = {}'.format(acc))

    lr = linear_model.LogisticRegression()
    lr.fit(train_x, train_y)
    pred_y = lr.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('sklearn acc = {}'.format(acc))