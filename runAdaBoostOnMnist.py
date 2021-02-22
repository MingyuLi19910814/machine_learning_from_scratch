from models.Adaboost import Adaboost
from tools.mnist_loader import DataProcessor
from sklearn.tree import DecisionTreeClassifier
import numpy as np


tree_num = 5
tree_height = 3


def evaluateWithBinaryAdaBoostScratch(train_x, train_y, test_x, test_y):
    classifier = Adaboost(DecisionTreeClassifier, tree_num, max_depth=tree_height)

    digit1 = 1
    digit2 = 2
    idx_train = np.bitwise_or(train_y == digit1, train_y == digit2)
    train_x = train_x[idx_train]
    train_y = train_y[idx_train]
    train_y[train_y == digit1] = 1
    train_y[train_y == digit2] = 0

    classifier.fit(train_x, train_y)

    idx_test = np.bitwise_or(test_y == digit1, test_y == digit2)
    test_x = test_x[idx_test]
    test_y = test_y[idx_test]
    test_y[test_y == digit1] = 1
    test_y[test_y == digit2] = 0
    score = classifier.score(test_x, test_y)
    print('AdaBoost classifier scores = {}, overall score = {}'.format(score[: -1], score[-1]))

if __name__ == "__main__":
    train_size = 60000
    processor = DataProcessor()
    train_x, train_y = processor.get()

    processor = DataProcessor(train='test')
    test_x, test_y = processor.get()

    evaluateWithBinaryAdaBoostScratch(train_x, train_y, test_x, test_y)