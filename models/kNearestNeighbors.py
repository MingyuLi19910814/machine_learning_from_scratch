import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class KNearestNeighbors:
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("inconsistency in X and Y size")
        if self.X.shape[0] == 0:
            raise ValueError("input sample can not be empty")
        if self.X.ndim == 1:
            self.X = np.reshape(self.X, (-1, 1))

    def query(self, xs, k):
        if k > self.X.shape[0]:
            raise ValueError('k is larger than number of data sample')
        elif k == 0:
            raise ValueError("k can not be zero")
        xs = np.asarray(xs)
        if xs.shape[0] == 0:
            raise ValueError("query can not be empty")
        ans = np.zeros((xs.__len__(), k), dtype=self.Y.dtype)
        for idx, x in enumerate(xs):
            dists = np.linalg.norm(x - self.X, axis=-1)
            sortedIdx = np.argsort(dists)[:k]
            ans[idx] = self.Y[sortedIdx]
        return ans


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)
    y_pred = KNearestNeighbors(train_x, train_y).query(test_x, 1)
    accuracy = accuracy_score(test_y, y_pred)
