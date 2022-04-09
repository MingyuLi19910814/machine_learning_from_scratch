import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from Evaluate import evaluate_classifier

class NaiveBayes:
    def __init__(self):
        self.encoder = LabelEncoder()

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        if X.ndim != 2 or X.shape[0] != Y.shape[0] or Y.ndim != 1:
            raise ValueError('invalid input')
        self.encoder.fit(Y)
        y_encoded = self.encoder.transform(Y)
        y_unique = np.unique(y_encoded)
        self.prior = np.zeros(y_unique.shape[0])
        self.mean = np.zeros(shape=(self.prior.shape[0], X.shape[1]))
        self.std = np.zeros(shape=(self.prior.shape[0], X.shape[1]))
        for y in y_unique:
            idx = y == y_encoded
            self.prior[y] = np.mean(idx)
            self.mean[y] = np.mean(X[idx], axis=0)
            self.std[y] = np.std(X[idx], axis=0)

    def predict(self, X):
        X = np.asarray(X)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError('invalid input')
        ans = np.zeros(X.shape[0], dtype=int)
        for idx, x in enumerate(X):
            likelihood = np.exp(-0.5 * ((x - self.mean) / self.std) ** 2) / (self.std * np.sqrt(2 * np.pi)) + 1e-13
            p = np.sum(np.log(likelihood), axis=-1) + np.log(self.prior)
            ans[idx] = np.argmax(p)
        ans = self.encoder.inverse_transform(ans)
        return ans

if __name__ == "__main__":
    cls1 = NaiveBayes()
    cls2 = GaussianNB()
    evaluate_classifier(cls1, cls2)