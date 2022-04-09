import numpy as np
from sklearn import tree
from collections import Counter
from Evaluate import evaluate_classifier

class Node:
    def __init__(self, X, Y, height, max_height, max_features=None, loss='entropy'):
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError('invalid input')
        self.height = height
        assert loss in ['entropy', 'gini']
        self.max_features = max_features
        self.loss = loss
        self.feature_idx = None
        self.threshold = None
        self.label = None
        self.left = None
        self.right = None
        self.feature_dim = X.shape[1]
        self.epsilon = 1e-13
        self.__build__(X, Y, max_height)

    def __loss__(self, Y):
        p = np.array(list(Counter(Y).values())) / Y.shape[0]
        if self.loss == 'entropy':
            loss = -np.sum(p * np.log(p + self.epsilon))
        else:
            loss = 1 - np.sum(p ** 2)
        return loss

    def __select_majority__(self, Y):
        counter = Counter(Y)
        self.label = counter.most_common(1)[0][0]

    def __build__(self, X, Y, max_height):
        if self.height == max_height or np.unique(Y).shape[0] == 1:
            self.__select_majority__(Y)
            return
        best_loss = self.__loss__(Y)
        if self.max_features is None:
            feature_indexes = np.arange(self.feature_dim)
        else:
            feature_indexes = np.random.choice(self.feature_dim, self.max_features, replace=False)

        for feature_idx in feature_indexes:
            values = np.unique(X[:, feature_idx])
            for value in values:
                i1 = X[:, feature_idx] <= value
                i2 = X[:, feature_idx] > value
                y1 = Y[i1]
                y2 = Y[i2]
                w1 = np.sum(i1) / X.shape[0]
                w2 = np.sum(i2) / X.shape[0]
                loss = w1 * self.__loss__(y1) + w2 * self.__loss__(y2)
                if loss < best_loss:
                    best_loss = loss
                    self.threshold = value
                    self.feature_idx = feature_idx
        if self.threshold is None:
            self.__select_majority__(Y)
            return
        i1 = X[:, self.feature_idx] <= self.threshold
        i2 = X[:, self.feature_idx] > self.threshold
        self.left = Node(X[i1], Y[i1], self.height + 1, max_height, self.max_features, self.loss)
        self.right = Node(X[i2], Y[i2], self.height + 1, max_height, self.max_features, self.loss)

class DecisionTreeClassifier:
    def __init__(self, max_height, max_features=None, loss='entropy'):
        self.max_height = max_height
        self.loss = loss
        self.max_features = max_features

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError('invalid input')
        self.node = Node(X, Y, 1, self.max_height, self.max_features, self.loss)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.node.feature_dim:
            raise ValueError('invalid input')
        ans = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            node = self.node
            while node.label is None:
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            ans[idx] = node.label
        return ans

if __name__ == "__main__":
    cls1 = DecisionTreeClassifier(max_height=8, loss='entropy')
    cls2 = tree.DecisionTreeClassifier(max_depth=8)
    evaluate_classifier(cls1, cls2)