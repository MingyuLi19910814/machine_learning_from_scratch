import numpy as np
from sklearn.metrics import *
from sklearn import neighbors
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class KNearestNeighbor:
    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y)
        if self.X.ndim != 2 or self.X.shape[0] != self.Y.shape[0] or self.Y.ndim != 1:
            raise ValueError("invalid input")

    def predict(self, X, k):
        ans = np.zeros(shape=(X.shape[0], k))
        for idx, x in enumerate(X):
            dists = np.linalg.norm(x - self.X, axis=-1)
            index = np.argsort(dists)[:k]
            ans[idx] = self.Y[index]
        return ans

if __name__ == "__main__":
    n_samples = 1000
    n_features = 10
    n_informative = 8
    n_classes = 5
    X, Y = make_classification(n_samples=n_samples,
                               n_features=n_features,
                               n_informative=n_informative,
                               n_classes=n_classes,
                               random_state=0)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=0)
    knn = KNearestNeighbor()
    knn.fit(train_x, train_y)
    pred_y = knn.predict(test_x, 1)
    acc = accuracy_score(test_y, pred_y)
    print('acc = {}'.format(acc))

    knn = neighbors.KNeighborsClassifier(n_neighbors=1)
    knn.fit(train_x, train_y)
    pred_y = knn.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('sklearn acc = {}'.format(acc))