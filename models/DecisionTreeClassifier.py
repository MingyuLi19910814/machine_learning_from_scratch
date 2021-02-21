import numpy as np
from tqdm import tqdm
from collections import Counter

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

    def __init__(self):
        self.root = None
        self.sample_dimension = None
        self.bar = None

    """
    get the entropy of using sum(-p*log(p))
    """
    @staticmethod
    def __get_entropy__(y):
        y_counter = Counter(y)
        p = np.asarray(list(y_counter.values()), dtype=np.float32)
        p /= y.shape[0]
        entropy = np.sum(-p * np.log(p))
        return entropy

    """
    build the subtree from sample(x) and label(y)
    """
    def __build__(self, x, y):
        # if all samples are from the same category, this should be a leaf node
        if set(y).__len__() == 1:
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

            isSmaller = x[:, best_feature_idx] < best_threshold
            isLarger = x[:, best_feature_idx] >= best_threshold
            x_small = x[isSmaller]
            y_small = y[isSmaller]

            x_large = x[isLarger]
            y_large = y[isLarger]

            # recursively build the child subtree
            node.left = self.__build__(x_small, y_small)
            node.right = self.__build__(x_large, y_large)
            return node

    def fit(self, x, y):
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
