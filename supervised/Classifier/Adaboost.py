import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from Evaluate import evaluate_classifier

class AdaBoost:
    def __init__(self, n_estimators, max_depth=1):
        self.n_estimators = n_estimators
        self.estimators = [DecisionTreeClassifier(max_depth=max_depth) for _ in range(self.n_estimators)]
        self.alpha = np.zeros(self.n_estimators)
        self.polarity = np.ones(self.n_estimators)

    def fit(self, X, Y):
        epsilon = 1e-7
        X = np.asarray(X)
        # we are changing Y, so need to copy it
        Y = np.array(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1 or sorted(np.unique(Y)) != [0, 1]:
            raise ValueError('invalid input')
        Y[Y == 0] = -1
        weights = np.ones(X.shape[0]) / X.shape[0]
        for idx in range(self.n_estimators):
            self.estimators[idx].fit(X, Y, sample_weight=weights)
            pred = self.estimators[idx].predict(X)
            error_sum = np.sum(weights[pred != Y]) / np.sum(weights)
            if error_sum > 0.5:
                self.polarity[idx] = -1
                error_sum = 1 - error_sum
                pred = 1 - pred
            self.alpha[idx] = 0.5 * np.log((1 - error_sum) / (error_sum + epsilon))

            weights = weights * np.exp(-self.alpha[idx] * Y * pred)
            weights = weights / np.sum(weights)

    def predict(self, X):
        X = np.asarray(X)
        prediction = np.zeros(X.shape[0])
        for idx in range(self.n_estimators):
            prediction += self.estimators[idx].predict(X) * self.polarity[idx] * self.alpha[idx]
        prediction[prediction >= 0] = 1
        prediction[prediction < 0] = 0
        return prediction

if __name__ == "__main__":
    n_estimators = 101
    cls1 = AdaBoost(max_depth=1, n_estimators=n_estimators)
    cls2 = ensemble.AdaBoostClassifier(n_estimators=n_estimators)
    evaluate_classifier(cls1, cls2)