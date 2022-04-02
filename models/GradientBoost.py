import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


class GradientBoostRegressor:
    def __init__(self, lr, max_it):
        self.lr = lr
        self.max_it = max_it
        self.initial_pred = None

    def __loss__(self, pred, gt):
        return 0.5 * np.mean((pred - gt) ** 2)

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        assert X.ndim == 2 and Y.ndim == 1 and X.shape[0] == Y.shape[0] and X.shape[0] != 0
        self.initial_pred = np.mean(Y)
        pred = np.ones_like(Y) * self.initial_pred
        self.trees = []
        losses = []
        for _ in range(self.max_it):
            res = Y - pred
            losses.append(self.__loss__(Y, pred))
            tree = DecisionTreeRegressor(max_depth=2, min_samples_split=5, min_samples_leaf=5, max_features=3)
            tree.fit(X, res)
            pred = pred + tree.predict(X) * self.lr
            self.trees.append(tree)
        plt.plot(losses)
        plt.show()

    def predict(self, X):
        X = np.asarray(X)
        ans = np.ones(X.shape[0]) * self.initial_pred
        for tree in self.trees:
            res = tree.predict(X)
            ans += res * self.lr
        return ans

if __name__ == "__main__":
    n_estimators = 1001  # No of boosting steps
    alpha = 0.01  # Learning rate
    input_ads = pd.read_csv('../test/housing.csv')
    X = input_ads[[cols for cols in list(input_ads.columns) if 'MEDV' not in cols]]
    y = input_ads['MEDV']

    train_x, test_x, train_y, test_y = train_test_split(X, y, test_size=0.30, random_state=100)
    scaler = StandardScaler()

    train_x = scaler.fit_transform(train_x)
    test_x = scaler.fit_transform(test_x)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    tree = GradientBoostRegressor(alpha, n_estimators)
    tree.fit(train_x, train_y)
    pred_y = tree.predict(test_x)


    sklearn_tree = GradientBoostingRegressor(n_estimators=n_estimators,
                                             max_depth=2,
                                             min_samples_split=5,
                                             min_samples_leaf=5,
                                             max_features=3,
                                             )
    sklearn_tree.fit(train_x, train_y)
    sklearn_pred = sklearn_tree.predict(test_x)
    print('sklearn rmse = {}'.format(np.sqrt(mean_squared_error(test_y, sklearn_pred))))