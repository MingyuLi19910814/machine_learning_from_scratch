import numpy as np
from collections import Counter

class Node:
    def __init__(self, X, Y, currentHeight, maxHeight):
        self.H = currentHeight
        self.label = None
        self.threshold = None
        self.featureIdx = None
        self.__build__(X, Y, maxHeight)
    @staticmethod
    def __entropy__(Y):
        p = np.asarray(list(Counter(Y).values()), dtype=float) / Y.shape[0]
        return -np.sum(p * np.log(p))

    @staticmethod
    def __split__(X, Y, featureIdx, threshold):
        Y1 = Y[X[:, featureIdx] < threshold]
        Y2 = Y[X[:, featureIdx] >= threshold]
        return Node.__entropy__(Y1) + Node.__entropy__(Y2)

    def __build__(self, X, Y, maxHeight):
        if np.unique(Y).shape[0] == 1:
            self.label = Y[0]
            return
        elif self.H == maxHeight:
            self.label = Counter(Y).most_common(1)[0][0]
            return

        bestEntropy = Node.__entropy__(Y)

        for featureIdx in range(X.shape[1]):
            for threshold in np.unique(X[:, featureIdx]):
                entropy = Node.__split__(X, Y, featureIdx, threshold)
                if entropy < bestEntropy:
                    bestEntropy = entropy
                    self.featureIdx = featureIdx
                    self.threshold = threshold

        if self.featureIdx is None:
            self.label = Counter(Y).most_common(1)[0][0]
            return
        X1 = X[X[:, self.featureIdx] < self.threshold]
        Y1 = Y[X[:, self.featureIdx] < self.threshold]
        X2 = X[X[:, self.featureIdx] >= self.threshold]
        Y2 = Y[X[:, self.featureIdx] >= self.threshold]
        self.left = None if X1.shape[0] == 0 else Node(X1, Y1, self.H + 1, maxHeight)
        self.right = None if X2.shape[0] == 0 else Node(X2, Y2, self.H + 1, maxHeight)

class DecisionTree:
    def __init__(self, X, Y, maxHeight=-1):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim == 1:
            X = X.reshape((-1, 1))
        if Y.ndim != 1 or X.shape[0] != Y.shape[0]:
            raise ValueError("input error")
        self.featureDim = X.shape[1]
        self.node = Node(X, Y, 0, maxHeight)
    def predict(self, X):
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape((1, -1))
        if X.shape[1] != self.featureDim:
            raise ValueError("input error")

        ans = []
        for x in X:
            node = self.node
            while node.label is None:
                node = node.left if x[node.featureIdx] < node.threshold else node.right
            ans.append(node.label)
        return np.asarray(ans)
if __name__ == "__main__":
    X = np.arange(0, 100)
    Y = np.concatenate([np.zeros(50), np.ones(50)], axis=0)
    tree = DecisionTree(X, Y, 3)
    print(tree.predict(np.arange(25, 75).reshape((-1, 1))))