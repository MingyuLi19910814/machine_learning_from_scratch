from tools.mnist_loader import  DataProcessor
from models.DecisionTreeModels import DecisionTree
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
    classifier.train(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print('DecisionTreeScratch score = {}'.format(score))


if __name__ == "__main__":
    train_size = 60000
    interval = 128

    processor = DataProcessor(interval=interval)
    train_x, train_y = processor.get()

    processor = DataProcessor(train='test', interval=interval)
    test_x, test_y = processor.get()

    evaluateWithSKLearn(train_x, train_y, test_x, test_y)
