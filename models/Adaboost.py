from models.ClassifierInterface import *
import numpy as np


class Adaboost(BinaryClassifier):
    class DecisionStump:
        def __init__(self):
            self.polarity = 1
            self.alpha = 0.0

    def __init__(self, base_classifier, num_classifier, **args):
        self.num_classifier = num_classifier
        self.weight = None
        self.classifiers = [base_classifier(**args) for _ in range(self.num_classifier)]
        self.stumps = [Adaboost.DecisionStump() for _ in range(self.num_classifier)]

    def fit(self, x, y):
        self.weight = np.ones(shape=(x.shape[0])) / x.shape[0]
        for clf_idx in range(self.num_classifier):
            print('start training {}/{} classifier'.format(clf_idx + 1, self.num_classifier))
            self.classifiers[clf_idx].fit(x, y, self.weight)
            prediction = self.classifiers[clf_idx].predict(x)
            errorIdx = prediction != y
            errorSum = np.sum(self.weight[errorIdx])

            if errorSum > 0.5:
                errorSum = 1 - errorSum
                self.stumps[clf_idx].polarity = -1
                prediction = 1 - prediction
            # update the weight of every classifier
            self.stumps[clf_idx].alpha = 0.5 * np.log((1.0 - errorSum) / (errorSum + 1e-10))
            # update the weight of each sample
            y_prediction = np.ones_like(y)
            y_prediction[y != prediction] = -1
            self.weight = self.weight * np.exp(-self.stumps[clf_idx].alpha * y_prediction)
            self.weight = self.weight / np.sum(self.weight)

    def predict(self, x):
        prediction = np.zeros(x.shape[0])

        for clf_idx in range(self.num_classifier):
            pred = self.classifiers[clf_idx].predict(x)
            pred[pred == 0] = -1
            if self.stumps[clf_idx].polarity == -1:
                pred = pred * -1
            prediction += pred * self.stumps[clf_idx].alpha
        return prediction > 0

    def score(self, x, y):
        classifier_score = []
        for classifier in self.classifiers:
            prediction = classifier.predict(x)
            prediction = prediction == y
            classifier_score.append(np.sum(prediction) / prediction.shape[0])
        prediction = self.predict(x)
        prediction = prediction == y
        classifier_score.append(np.sum(prediction) / prediction.shape[0])
        return classifier_score
