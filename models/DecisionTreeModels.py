import numpy as np
from tqdm import tqdm


class DecisionTree:
    class Node:
        """
        node in the decision tree
        """
        def __init__(self):
            # this node uses the feature_idxth feature for dividing
            self.feature_idx = -1
            # if this node is a leaf, the label will be the saved, otherwise it's None
            self.label = None
            # the threshold for dividing the subtree
            self.threshold = None
            # the left child, stores the subtree built from the values smaller than the threshold
            self.left = None
            # the right node, stores the subtree built from the values larger than the threshold
            self.right = None

    def __init__(self, algorithm='ID3'):
        self.model = algorithm
        self.root = None
        self.sample_dimension = None
        self.bar = None
        assert self.model in {'ID3', 'C45', 'CART'}

    """
    get the entropy of using sum(-p*log(p))
    """
    @staticmethod
    def __get_entropy__(y):
        entropy = 0.0
        for y_val in set(y):
            p = np.sum(y == y_val) * 1.0 / y.shape[0]
            entropy -= p * np.log(p)
        return entropy

    """
    build the subtree from sample(x) and label(y)
    """
    def __build__(self, x, y):
        y_val_set = set(y)
        # if all samples are from the same category, this should be a leaf node
        if y_val_set.__len__() == 1:
            node = DecisionTree.Node()
            node.label = y[0]
            self.bar.update(y.shape[0])
            return node
        else:
            current_entropy = DecisionTree.__get_entropy__(y)
            best_entropy = current_entropy
            best_feature_idx = -1
            best_threshold = None
            # loop of every feature
            for feature_idx in range(x.shape[1]):
                x_feature_idx = x[:, feature_idx]
                x_feature_idx_sorted = sorted(set(x_feature_idx))
                # to handle the continuous values of the feature, the feature values are sorted in increasing order
                # and the median of every two continuous values is computed. Try to divide the subtree with this median
                # and get the entropy gain.
                for i in range(x_feature_idx_sorted.__len__() - 1):
                    v1 = float(x_feature_idx_sorted[i])
                    v2 = float(x_feature_idx_sorted[i + 1])
                    threshold = (v1 + v2) * 0.5
                    y_small = y[x_feature_idx < threshold]
                    y_large = y[x_feature_idx >= threshold]
                    new_entropy = DecisionTree.__get_entropy__(y_small) * y_small.shape[0] / y.shape[0]
                    new_entropy += DecisionTree.__get_entropy__(y_large) * y_large.shape[0] / y.shape[0]
                    # the feature index and best dividing threshold are kept
                    if new_entropy < best_entropy:
                        best_entropy = new_entropy
                        best_feature_idx = feature_idx
                        best_threshold = threshold

            node = DecisionTree.Node()
            node.feature_idx = best_feature_idx
            node.threshold = best_threshold

            x_small = x[x[:, best_feature_idx] < best_threshold]
            y_small = y[x[:, best_feature_idx] < best_threshold]

            x_large = x[x[:, best_feature_idx] >= best_threshold]
            y_large = y[x[:, best_feature_idx] >= best_threshold]
            # recursively build the child subtree
            node.left = self.__build__(x_small, y_small)
            node.right = self.__build__(x_large, y_large)
            return node

    def train(self, x, y):
        self.sample_dimension = x.shape[1]
        self.bar = tqdm(total=x.shape[0])
        assert len(x.shape) == 2 and len(y.shape) == 1 and x.shape[0] == y.shape[0]
        self.root = self.__build__(x, y)
        self.bar.close()

    def predict(self, xs):
        # the predict sample dimension must be the same as train samples
        assert xs.shape[1] == self.sample_dimension
        ys = [self.__predictOneSample__(x) for x in xs]
        return np.asarray(ys)

    def __predictOneSample__(self, x):
        node = self.root
        while node.label is None:
            feature_idx = node.feature_idx
            feature_val = x[feature_idx]
            node = node.left if feature_val < node.threshold else node.right
        return node.label

    def score(self, x, y):
        y_predict = self.predict(x)
        return np.sum(y_predict == y) * 1.0 / y.shape[0]
