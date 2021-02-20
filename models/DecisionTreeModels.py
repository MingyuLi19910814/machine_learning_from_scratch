import numpy as np
from tqdm import tqdm


class DecisionTree:

    class Node:
        def __init__(self):
            self.feature_idx = -1
            self.label = None
            self.children = {}
            self.sample_cnt = 0

    def __init__(self, algorithm='ID3'):
        self.model = algorithm
        self.root = None
        self.sample_dimension = None
        self.bar = None
        self.label_cnt = {}
        assert self.model in {'ID3', 'C45', 'CART'}

    @staticmethod
    def __get_entropy__(y):
        entropy = 0.0
        for y_val in set(y):
            p = np.sum(y == y_val) * 1.0 / y.shape[0]
            entropy -= p * np.log(p)
        return entropy

    def __build__(self, x, y):
        y_val_set = set(y)
        if y_val_set.__len__() == 1:
            node = DecisionTree.Node()
            node.label = y[0]
            node.sample_cnt = y.shape[0]
            self.bar.update(y.shape[0])
            return node
        else:
            current_entropy = DecisionTree.__get_entropy__(y)
            best_entropy = current_entropy
            best_feature_idx = -1
            for feature_idx in range(x.shape[1]):
                x_feature_idx = x[:, feature_idx]
                new_entropy = 0.0
                for x_val in set(x_feature_idx):
                    y_val = y[x_feature_idx == x_val]
                    new_entropy += DecisionTree.__get_entropy__(y_val) * y_val.shape[0] / y.shape[0]
                if new_entropy < best_entropy:
                    best_entropy = new_entropy
                    best_feature_idx = feature_idx

            node = DecisionTree.Node()
            node.feature_idx = best_feature_idx
            x_best_feature_idx = x[:, best_feature_idx]
            for x_val in set(x_best_feature_idx):
                node.children[x_val] = self.__build__(x[x_best_feature_idx == x_val], y[x_best_feature_idx == x_val])
            return node

    def train(self, x, y):
        self.sample_dimension = x.shape[1]
        self.bar = tqdm(total=x.shape[0])
        assert len(x.shape) == 2 and len(y.shape) == 1 and x.shape[0] == y.shape[0]
        self.root = self.__build__(x, y)
        self.bar.close()

    def predict(self, xs):
        assert xs.shape[1] == self.sample_dimension
        ys = [self.__predictOneSample__(x) for x in xs]
        return np.asarray(ys)

    def __get_label_cnt__(self, node):
        if node.label is not None:
            self.label_cnt[node.label] = self.label_cnt.get(node.label, 0) + node.sample_cnt
        else:
            for child in node.children.values():
                self.__get_label_cnt__(child)

    def __predictOneSample__(self, x):
        node = self.root
        self.label_cnt = {}
        while node.label is None:
            feature_idx = node.feature_idx
            feature_val = x[feature_idx]
            if feature_val in node.children:
                node = node.children[feature_val]
            else:
                self.__get_label_cnt__(node)
                max_cnt = 0
                best_label = None
                for label, cnt in self.label_cnt.items():
                    if cnt > max_cnt:
                        max_cnt = cnt
                        best_label = label
                return best_label
        return node.label

    def score(self, x, y):
        y_predict = self.predict(x)
        return np.sum(y_predict == y) * 1.0 / y.shape[0]