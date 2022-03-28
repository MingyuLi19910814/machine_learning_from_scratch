import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from collections import Counter


class BaseNode:
    def __init__(self, X, Y, height, max_height):
        self.leaf_value = None
        self.feature_idx = None
        self.feature_dim = X.shape[1]
        self.threshold = None
        self.left = None
        self.right = None
        self.__build__(X, Y, height, max_height)

    def __createLeafNode__(self, X, Y):
        raise NotImplementedError()

    def __calc__(self, Y):
        raise NotImplementedError()

    def __create_child__(self, X, Y, height, max_height):
        raise NotImplementedError()

    def __build__(self, X, Y, height, max_height):
        if height == max_height or np.unique(Y).size == 1:
            self.__createLeafNode__(X, Y)
            return

        best_v = self.__calc__(Y)
        for feature_idx in range(self.feature_dim):
            feature_values = np.unique(X[:, feature_idx])
            for feature_value in feature_values:
                i1 = X[:, feature_idx] < feature_value
                i2 = X[:, feature_idx] >= feature_value
                v1 = self.__calc__(Y[i1])
                v2 = self.__calc__(Y[i2])
                v = (np.sum(i1) * v1 + np.sum(i2) * v2) / X.shape[0]
                if v < best_v:
                    best_v = v
                    self.feature_idx = feature_idx
                    self.threshold = feature_value
        if self.feature_idx is None:
            self.__createLeafNode__(X, Y)
            return
        i1 = X[:, self.feature_idx] < self.threshold
        i2 = X[:, self.feature_idx] >= self.threshold
        self.left = self.__create_child__(X[i1], Y[i1], height + 1, max_height)
        self.right = self.__create_child__(X[i2], Y[i2], height + 1, max_height)

class EntropyNode(BaseNode):
    def __createLeafNode__(self, X, Y):
        self.leaf_value = Counter(Y).most_common(1)[0][0]

    def __calc__(self, Y):
        p = np.array(list(Counter(Y).values())).astype(float) / Y.shape[0]
        return np.sum(-p * np.log(p + 1e-13))

    def __create_child__(self, X, Y, height, max_height):
        return EntropyNode(X, Y, height, max_height)

class GiniNode(BaseNode):
    def __createLeafNode__(self, X, Y):
        self.leaf_value = Counter(Y).most_common(1)[0][0]

    def __calc__(self, Y):
        p = np.array(list(Counter(Y).values())).astype(float) / Y.shape[0]
        return 1 - np.sum(p * p)

    def __create_child__(self, X, Y, height, max_height):
        return GiniNode(X, Y, height, max_height)

class DecisionTree:
    def __init__(self, X, Y, max_height=-1):
        X = np.asarray(X).astype(float)
        Y = np.asarray(Y).astype(int)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("different length of X and Y!")
        elif X.shape[0] == 0:
            raise ValueError("train X must not be empty")
        elif X.ndim != 2:
            raise ValueError("X must be 2-dimensional")
        elif Y.ndim != 1:
            raise ValueError("Y must be 1-dimensional")
        self.node = GiniNode(X, Y, 0, max_height)

    def predict(self, X):
        X = np.asarray(X).astype(float)
        X = X.reshape((-1, self.node.feature_dim))
        ans = []
        for x in X:
            node = self.node
            while node.leaf_value is None:
                feature_idx = node.feature_idx
                threshold = node.threshold
                node = node.left if x[feature_idx] < threshold else node.right
            ans.append(node.leaf_value)
        return np.array(ans)


if __name__ == "__main__":
    dataset = load_iris()
    X = dataset.data
    Y = dataset.target
    acc = []
    for random_state in range(43):
        train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25, random_state=random_state)
        tree = DecisionTree(train_x, train_y)
        pred_y = tree.predict(test_x)
        accuracy = accuracy_score(test_y, pred_y)
        acc.append(accuracy)
        print('state = {}, acc = {}'.format(random_state, accuracy))
    print('average acc = {}'.format(np.mean(acc)))
