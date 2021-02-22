import numpy as np
from models.DecisionTreeClassifier import DecisionTree
from collections import Counter
from models.ClassifierInterface import *


class RandomForest(MultiClassClassifier):
    def __init__(self, tree_num):
        self.tree_num = tree_num
        self.trees = []

    def fit(self, x, y):
        self.trees = []
        for _ in range(self.tree_num):
            tree = DecisionTree()
            sample_idx = np.random.choice(np.arange(x.shape[0]), size=x.shape[0], replace=True)
            tree.fit(x[sample_idx], y[sample_idx])
            self.trees.append(tree)

    def predict(self, xs):
        return np.asarray([self.__predictOneSample__(x) for x in xs])

    def score(self, x, y):
        predict_y = self.predict(x)
        return np.sum(y == predict_y) * 1.0 / y.shape[0]

    def __predictOneSample__(self, x):
        predictions = [tree.__predictOneSample__(x) for tree in self.trees]
        counter = Counter(predictions)
        return counter.most_common(1)[0][0]