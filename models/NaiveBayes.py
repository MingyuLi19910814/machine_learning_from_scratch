import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayes:
    def __init__(self):
        self.prior = None
        self.mean = None
        self.std = None
        self.class_num = None
        self.feature_dim = None

    def fit(self, X, Y):
        self.class_num = np.unique(Y).shape[0]
        X = np.asarray(X)
        Y = np.asarray(Y).astype(int)
        self.feature_dim = X.shape[1]
        self.prior = np.zeros(self.class_num)
        self.mean = np.zeros((self.class_num, self.feature_dim))
        self.std = np.zeros_like(self.mean)
        for class_idx in range(self.class_num):
            features = X[Y == class_idx]
            self.prior[class_idx] = features.shape[0] / X.shape[0]
            self.mean[class_idx] = np.mean(features, axis=0)
            self.std[class_idx] = np.std(features, axis=0)

    def predict(self, X):
        ans = []
        epsilon = 1e-13
        for x in X:
            probs = []
            for class_idx in range(self.class_num):
                prob = np.sum(np.log(self.__gaussian_density(x, class_idx) + epsilon)) + np.log(self.prior[class_idx] + epsilon)
                probs.append(prob)
            ans.append(np.argmax(probs))
        return ans

    def __gaussian_density(self, x, class_idx):
        mean = self.mean[class_idx]
        std = self.std[class_idx]
        density = np.exp(-0.5 * ((x - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
        return density


if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)
    classifier = NaiveBayes()
    classifier.fit(train_x, train_y)
    pred_y = classifier.predict(test_x)
    accuracy = accuracy_score(test_y, pred_y)
    print(accuracy)
