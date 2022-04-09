from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_rmse(regressor, train_x, test_x, train_y, test_y):
    regressor.fit(train_x, train_y)
    pred = regressor.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    return rmse

def evaluate_regressor(reg1, reg2, n_samples=1000, n_features=10, noise=1.0):
    X, Y = make_regression(n_samples=n_samples, n_features=n_features, n_informative=n_features, random_state=0, noise=noise)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    err1 = evaluate_rmse(reg1, train_x, test_x, train_y, test_y)
    err2 = evaluate_rmse(reg2, train_x, test_x, train_y, test_y)
    print('err1 = {}, err2 = {}'.format(err1, err2))