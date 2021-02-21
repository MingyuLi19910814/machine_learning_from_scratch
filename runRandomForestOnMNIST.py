from models.RandomForestClassifier import RandomForest
from tools.mnist_loader import DataProcessor
from sklearn.ensemble import RandomForestClassifier

treeNum = 5


def evaluateWithSKLearn(train_x, train_y, test_x, test_y):
    estimator = RandomForestClassifier(criterion='entropy', n_estimators=treeNum)
    print('start fit sklearn')
    estimator.fit(train_x, train_y)
    accuracy = estimator.score(test_x, test_y)
    print('id3 package score = {}'.format(accuracy))


def evaluateWithRandomForestScratch(train_x, train_y, test_x, test_y):
    classifier = RandomForest(treeNum)
    classifier.fit(train_x, train_y)
    score = classifier.score(test_x, test_y)
    print('DecisionTreeScratch score = {}'.format(score))


if __name__ == "__main__":
    processor = DataProcessor()
    train_x, train_y = processor.get()

    processor = DataProcessor(train='test')
    test_x, test_y = processor.get()

    evaluateWithSKLearn(train_x, train_y, test_x, test_y)
