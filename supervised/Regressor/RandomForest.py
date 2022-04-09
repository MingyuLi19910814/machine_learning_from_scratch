from DecisionTree import DecisionTree
import numpy as np
from Evaluate import evaluate_regressor

class RandomForest:
    def __init__(self, max_features=None, n_estimators=1, max_depth=-1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.trees = [DecisionTree(max_features=max_features, max_height=max_depth) for i in range(self.n_estimators)]

    def fit(self, X, Y):
        for i in range(self.n_estimators):
            idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
            self.trees[i].fit(X[idx], Y[idx])

    def predict(self, X):
        ans = np.zeros(X.shape[0])
        for i, x in enumerate(X):
            value = np.mean([tree.predict(x.reshape((1, -1)))[0] for tree in self.trees])
            ans[i] = value
        return ans

if __name__ == "__main__":
    n_estimators = 5
    max_depth = 5
    reg1 = RandomForest(max_features=None, n_estimators=n_estimators, max_depth=max_depth)
    reg2 = RandomForest(n_estimators=n_estimators, max_depth=max_depth, max_features=None)
    evaluate_regressor(reg1, reg2)

