from models.LogisticRegressionClassifier import BinaryLogisticRegression
from tools.mnist_loader import DataProcessor
import numpy as np
# def evaluateWithSKLearn(train_x, train_y, test_x, test_y):
#     estimator = DecisionTreeClassifier(criterion='entropy')
#     print('start fit sklearn')
#     estimator.fit(train_x, train_y)
#     estimator.predict(test_x)
#     accuracy = estimator.score(test_x, test_y)
#     print('id3 package score = {}'.format(accuracy))


def evaluateWithLogisticRegressionScratch(train_x, train_y, test_x, test_y):
    classifier = BinaryLogisticRegression(max_iter=100)
    classifier.fit(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print('DecisionTreeScratch score = {}'.format(score))


if __name__ == "__main__":
    processor = DataProcessor(interval=1)
    train_x, train_y = processor.get()
    i1 = train_y == 1
    i2 = train_y == 2
    idx = np.bitwise_or(i1, i2)
    train_x = train_x[idx] / 255.0
    train_y = train_y[idx] == 2
    print('train size = {}, sum = {}'.format(train_y.shape, np.sum(train_y)))

    processor = DataProcessor(train='test', interval=1)
    test_x, test_y = processor.get()
    i1 = test_y == 1
    i2 = test_y == 2
    idx = np.bitwise_or(i1, i2)
    test_x = test_x[idx] / 255.0
    test_y = test_y[idx] == 2
    print('test size = {}, sum = {}'.format(test_y.shape, np.sum(test_y)))

    evaluateWithLogisticRegressionScratch(train_x, train_y, test_x, test_y)