import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, KFold
from collections import Counter
from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier

class Node:
    def __init__(self, X, Y, current_height, max_height=-1, criteria='gini'):
        '''
        X: 2d np.ndarray
        Y: 1d np.ndarray
        '''
        self.label = None
        self.feature_dim = X.shape[1]
        self.feature_idx = None
        self.threshold = None
        self.height = current_height
        self.criteria = criteria
        self.left = None
        self.right = None
        assert criteria in ['gini', 'entropy']
        self.__build__(X, Y, max_height)

    def __calc__(self, pred):
        p = np.asarray(list(Counter(pred).values())) / pred.shape[0]
        if self.criteria == 'gini':
            return 1 - np.sum(p ** 2)
        else:
            return -np.sum(p * np.log(p + 1e-13))

    def __decide_majority__(self, Y):
        return Counter(Y).most_common(1)[0][0]

    def __build__(self, X, Y, max_height):
        if self.height == max_height:
            self.label = self.__decide_majority__(Y)
            return
        best_cost = self.__calc__(Y)
        for feature_idx in range(self.feature_dim):
            feature_values = np.unique(X[:, feature_idx])
            for feature_value in feature_values:
                i1 = X[:, feature_idx] <= feature_value
                i2 = X[:, feature_idx] > feature_value
                y1 = Y[i1]
                y2 = Y[i2]
                w1 = np.sum(i1).astype(float) / i1.shape[0]
                w2 = np.sum(i2).astype(float) / i1.shape[0]
                cost = w1 * self.__calc__(y1) + w2 * self.__calc__(y2)
                if cost < best_cost:
                    self.feature_idx = feature_idx
                    self.threshold = feature_value
                    best_cost = cost
        if self.threshold is None:
            self.label = self.__decide_majority__(Y)
            return
        i1 = X[:, self.feature_idx] <= self.threshold
        i2 = X[:, self.feature_idx] > self.threshold
        x1 = X[i1]
        x2 = X[i2]
        y1 = Y[i1]
        y2 = Y[i2]
        self.left = Node(x1, y1, self.height + 1, max_height, self.criteria)
        self.right = Node(x2, y2, self.height + 1, max_height, self.criteria)

class DecisionTree:
    def __init__(self, max_height=-1, criteria='gini'):
        self.max_height = max_height
        self.criteria = criteria

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or X.shape[0] == 0:
            raise ValueError("invalid input")
        self.node = Node(X, Y, 1, self.max_height, self.criteria)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.node.feature_dim:
            raise ValueError("invalid input")
        res = []
        for x in X:
            node = self.node
            while node.label is None:
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            res.append(node.label)
        return np.array(res)


if __name__ == "__main__":
    n_samples = 2000
    n_classes = 2
    max_height = 3
    n_features = 5
    n_informative = 2
    X, Y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=0)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    cls = DecisionTree(max_height=max_height)
    cls.fit(train_x, train_y)
    pred_y = cls.predict(test_x)
    f1 = f1_score(test_y, pred_y, average=None)
    print('implemented f1 = {}'.format(f1))

    cls = DecisionTreeClassifier(max_depth=max_height)
    cls.fit(train_x, train_y)
    pred_y = cls.predict(test_x)
    f1 = f1_score(test_y, pred_y, average=None)
    print('sklearn f1 = {}'.format(f1))