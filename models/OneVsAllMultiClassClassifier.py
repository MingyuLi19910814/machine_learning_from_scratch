from models.ClassifierInterface import *
import numpy as np
import copy


class OneVsAllMultiClassClassifier(MultiClassClassifier):
    def __init__(self, classifier, class_num):
        assert isinstance(classifier, ClassifierOutputPossibility)
        self.class_num = class_num
        self.classifiers = [copy.deepcopy(classifier) for i in range(self.class_num)]

    def fit(self, x, y):
        for i in range(self.class_num):
            labels = np.zeros_like(y)
            labels[y == i] = 1
            print('start training {}/{}th classifier'.format(i + 1, self.class_num))
            self.classifiers[i].fit(x, labels)

    def predict(self, x):
        predictions = [np.reshape(classifier.predictPossibility(x), (-1, 1)) for classifier in self.classifiers]
        predictions = np.hstack(predictions)
        predictions = np.argmax(predictions, axis=1)
        return predictions

    def score(self, x, y):
        prediction = self.predict(x)
        acc = np.sum(prediction == y) * 1.0 / y.shape[0]
        return acc
