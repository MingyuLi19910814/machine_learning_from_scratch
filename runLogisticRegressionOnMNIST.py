from models.LogisticRegressionClassifier import MultiClassLogisticRegression
from tools.mnist_loader import DataProcessor
from sklearn.linear_model import LogisticRegression

def evaluateWithSKLearn(train_x, train_y, test_x, test_y):
    estimator = LogisticRegression(penalty='none', max_iter=2500, multi_class='ovr', n_jobs=1)
    print('start fit sklearn')
    estimator.fit(train_x, train_y)
    estimator.predict(test_x)
    accuracy = estimator.score(test_x, test_y)
    print('sklearn logistic regression package score = {}'.format(accuracy))

def evaluateWithMultiClassRegressionScratch(train_x, train_y, test_x, test_y):
    classifier = MultiClassLogisticRegression(class_num=10, max_iter=2500)
    classifier.fit(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print('multi class logistic regression score = {}'.format(score))

if __name__ == "__main__":
    train_size = 60000
    processor = DataProcessor(interval=1)
    train_x, train_y = processor.get()
    train_x = train_x[:train_size, :] / 255.0
    train_y = train_y[:train_size]

    processor = DataProcessor(train='test', interval=1)
    test_x, test_y = processor.get()
    test_x = test_x / 255.0

    evaluateWithMultiClassRegressionScratch(train_x, train_y, test_x, test_y)