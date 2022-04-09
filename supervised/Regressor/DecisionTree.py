import numpy as np
from sklearn import tree
from Evaluate import evaluate_regressor

class Node:
    def __init__(self, X, Y, min_samples_leaf, height, max_features, max_height=-1):
        self.height = height
        self.threshold = None
        self.max_features = max_features
        self.feature_idx = None
        self.value = None
        self.left = None
        self.right = None
        self.__build__(X, Y, min_samples_leaf, max_height)

    def __cost__(self, Y):
        #return 0 if Y.shape[0] == 0 else np.var(Y)
        return 0 if Y.shape[0] == 0 else np.sum((Y - np.mean(Y)) ** 2)

    def __create_leaf__(self, Y):
        self.value = np.mean(Y)

    def __build__(self, X, Y, min_samples_leaf, max_height):
        if self.height == max_height or Y.shape[0] <= min_samples_leaf:
            self.__create_leaf__(Y)
            return

        best_cost = self.__cost__(Y)
        feature_indexes = np.arange(X.shape[1]) if self.max_features is None else np.random.choice(X.shape[1], self.max_features, replace=False)
        for feature_idx in feature_indexes:
            values = np.unique(X[:, feature_idx])
            for value in values:
                i1 = X[:, feature_idx] <= value
                i2 = X[:, feature_idx] > value
                w1 = np.sum(i1) / X.shape[0]
                w2 = np.sum(i2) / X.shape[0]
                cost = self.__cost__(Y[i1]) * w1 + self.__cost__(Y[i2]) * w2
                if cost < best_cost:
                    best_cost = cost
                    self.feature_idx = feature_idx
                    self.threshold = value
        if self.threshold is None:
            self.__create_leaf__(Y)
            return
        i1 = X[:, self.feature_idx] <= self.threshold
        i2 = X[:, self.feature_idx] > self.threshold
        self.left = Node(X[i1], Y[i1], min_samples_leaf, self.height+1, self.max_features, max_height)
        self.right = Node(X[i2], Y[i2], min_samples_leaf, self.height+1, self.max_features, max_height)

class DecisionTree:
    def __init__(self, max_features=None, min_samples_leaf=1, max_height=-1):
        self.min_samples_leaf = min_samples_leaf
        self.max_height = max_height
        self.node = None
        self.max_features = max_features
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError("invalid input")
        self.node = Node(X, Y, self.min_samples_leaf, 0, self.max_features, self.max_height)

    def predict(self, X):
        ans = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            node = self.node
            while node.value is None:
                node = node.left if x[node.feature_idx] <= node.threshold else node.right
            ans[idx] = node.value
        return ans


if __name__ == "__main__":
    max_depth = 32
    min_samples_leaf = 4
    reg1 = DecisionTree(min_samples_leaf, max_depth)
    reg2 = tree.DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=min_samples_leaf)
    evaluate_regressor(reg1, reg2)



