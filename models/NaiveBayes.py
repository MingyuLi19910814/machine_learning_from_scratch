import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB

class NaiveBayes:
    def __init__(self):
        self.scaler = StandardScaler()
        self.prior = None
        self.feature_num = None
        self.class_num = None
    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError("invalid input")
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(Y)
        y_encoded = self.encoder.transform(Y)
        self.feature_num = X.shape[1]
        self.class_num = self.encoder.classes_.__len__()
        self.mean = np.zeros(shape=(self.class_num, self.feature_num))
        self.std = np.zeros(shape=(self.class_num, self.feature_num))
        self.prior = np.zeros(shape=(self.class_num))
        for label in range(self.encoder.classes_.__len__()):
            idx = y_encoded == label
            self.prior[label] = np.mean(idx)
            self.mean[label] = np.mean(X[idx], axis=0)
            self.std[label] = np.std(X[idx], axis=0)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[1] != self.feature_num:
            raise ValueError("invalid input")
        likelihood = np.exp(-0.5 * ((X[:, np.newaxis,:] - self.mean) / self.std) ** 2) / (self.std * np.sqrt(2 * np.pi)) + 1e-13
        log_likelihood = np.sum(np.log(likelihood), axis=-1)
        p = log_likelihood + np.log(self.prior)
        ans = np.argmax(p, axis=-1)
        ans = self.encoder.inverse_transform(ans)
        return ans





if __name__ == "__main__":
    n_samples = 1000
    n_features = 4
    n_informative = 4
    n_redundant = 0
    n_classes = 2
    X, Y = datasets.make_classification(n_samples=n_samples,
                                        n_classes=n_classes,
                                        n_features=n_features,
                                        n_informative=n_informative,
                                        n_redundant=n_redundant)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2)
    cls = NaiveBayes()
    cls.fit(train_x, train_y)
    pred_y = cls.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('implemented acc = {}'.format(acc))

    cls = GaussianNB()
    cls.fit(train_x, train_y)
    pred_y = cls.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('sklearn acc = {}'.format(acc))