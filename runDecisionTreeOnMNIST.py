from tools.mnist_loader import DataProcessor
from models.DecisionTreeClassifier import DecisionTree
import numpy as np
from sklearn.tree import DecisionTreeClassifier

def evaluateWithSKLearn(train_x, train_y, test_x, test_y):
    estimator = DecisionTreeClassifier(criterion='entropy')
    print('start fit sklearn')
    estimator.fit(train_x, train_y)
    estimator.predict(test_x)
    accuracy = estimator.score(test_x, test_y)
    print('id3 package score = {}'.format(accuracy))


def evaluateWithDecisionTreeScratch(train_x, train_y, test_x, test_y):
    classifier = DecisionTree()
    classifier.fit(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print('DecisionTreeScratch score = {}'.format(score))


if __name__ == "__main__":
    processor = DataProcessor()
    train_x, train_y = processor.get()

    processor = DataProcessor(train='test')
    test_x, test_y = processor.get()

    evaluateWithDecisionTreeScratch(train_x, train_y, test_x, test_y)
