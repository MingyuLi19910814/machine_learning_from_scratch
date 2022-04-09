import numpy as np
from DecisionTree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from Evaluate import evaluate_classifier


class RandomForest:
    def __init__(self, n_estimators, max_depth, max_features):
        self.n_estimators = n_estimators
        self.trees = [DecisionTreeClassifier(max_depth, loss='gini', max_features=max_features) for _ in range(self.n_estimators)]
    def fit(self, X, Y):
        for i in range(self.n_estimators):
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            self.trees[i].fit(X[idx], Y[idx])

    def predict(self, X):
        ans = np.zeros(X.shape[0])
        for idx, x in enumerate(X):
            results = [tree.predict(x.reshape((1, -1)))[0] for tree in self.trees]
            ans[idx] = Counter(results).most_common(1)[0][0]
        return ans


if __name__ == "__main__":
    n_estimators = 5
    max_depth = 5
    cls1 = RandomForest(n_estimators, max_depth, max_features=None)
    cls2 = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, max_features=None)
    evaluate_classifier(cls1, cls2)