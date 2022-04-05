import numpy as np
from sklearn import preprocessing
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


def getDigits():
    digits = load_digits()
    X = digits.data
    Y = digits.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, random_state=0, test_size=0.2)
    return (train_x, train_y), (test_x, test_y)

def LabelEncoder():
    print('*' * 20 + ' LabelEncoder ' + '*' * 20)
    y1 = ['a', 'a', 'b', 'c', 'a', 'c', 'd']
    y2 = [0, 0, 1, 2, 0, 2, 3]
    encoder = preprocessing.LabelEncoder()
    encoder.fit(y1)
    print(encoder.transform(y1))
    print(encoder.inverse_transform(y2))
    print('*' * 54)

def OneHotEncoder():
    print('*' * 20 + ' LabelEncoder ' + '*' * 20)
    y1 = ['a', 'a', 'b', 'c', 'a', 'c', 'd']
    encoder = preprocessing.LabelBinarizer()
    encoder.fit(y1)
    y1_one_hot = encoder.transform(y1)
    print('y1 one hot = \n{}'.format(y1_one_hot))
    print('inverse = {}'.format(encoder.inverse_transform(y1_one_hot)))

def StandardScaler():
    print('*' * 20 + ' StandardScaler ' + '*' * 20)
    (train_x, train_y), (test_x, test_y) = getDigits()
    processor = preprocessing.StandardScaler()
    standardized_train_x = processor.fit_transform(train_x)
    standardized_train_x_2 = preprocessing.scale(train_x)
    for dim in range(train_x.shape[1]):
        print('dim = {}, mean = {}, std = {}'.format(dim, np.mean(standardized_train_x[:, dim]), np.std(standardized_train_x[:, dim])))
    assert np.array_equal(standardized_train_x, standardized_train_x_2)
    print('*' * 54)

def RobustScaler():
    print('*' * 20 + ' RobustScalar ' + '*' * 20)
    (train_x, train_y), (test_x, test_y) = getDigits()
    processor = preprocessing.RobustScaler()
    scaled_train_x = processor.fit_transform(train_x)
    for dim in range(train_x.shape[1]):
        print('dim = {}, mean = {}, std = {}'.format(dim, np.mean(scaled_train_x[:, dim]), np.std(scaled_train_x[:, dim])))
    print('*' * 54)

def MinMaxScale():
    print('*' * 20 + ' MinMaxScaler ' + '*' * 20)
    (train_x, train_y), (test_x, test_y) = getDigits()
    print('train x range = {}'.format((np.min(train_x), np.max(train_x))))
    processor = preprocessing.MinMaxScaler()
    scaled_train_x = processor.fit_transform(train_x)
    scaled_train_x_2 = preprocessing.minmax_scale(train_x)
    assert np.array_equal(scaled_train_x, scaled_train_x_2)
    print('scaled train x range = {}'.format((np.min(scaled_train_x), np.max(scaled_train_x))))
    print('*' * 54)


if __name__ == "__main__":
    #MinMaxScale()
    StandardScaler()
    RobustScaler()
    #LabelEncoder()
    #OneHotEncoder()

