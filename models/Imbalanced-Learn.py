import numpy as np
from sklearn.datasets import load_digits
from imblearn.over_sampling import RandomOverSampler, SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt

def SVM():
    print('*' * 20 + ' SVM ' + '*' * 20)
    n_sample = 5000
    n_feature = 52
    n_informative = 50
    n_classes = 2
    X, Y = datasets.make_classification(n_samples=n_sample,
                                        n_classes=n_classes,
                                        n_features=n_feature,
                                        weights=[0.1, 0.9],
                                        n_informative=n_informative,
                                        random_state=0)
    kfold = sklearn.model_selection.KFold(n_splits=5)
    for train_idx, test_idx in kfold.split(X):
        train_x = X[train_idx]
        train_y = Y[train_idx]
        test_x = X[test_idx]
        test_y = Y[test_idx]
        assert train_idx.shape[0] + test_idx.shape[0] == X.shape[0]
        cls = sklearn.svm.SVC(class_weight='balanced')
        cls.fit(train_x, train_y)
        pred_y = cls.predict(test_x)
        f1 = sklearn.metrics.f1_score(test_y, pred_y, average=None)
        print('f1 = {}'.format(f1))

if __name__ == "__main__":
    SVM()