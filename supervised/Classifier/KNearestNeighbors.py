import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
from Evaluate import evaluate_classifier

class KnearestNeighbors:
    def __init__(self):
        pass

    def fit(self, X, Y):
        self.X = np.array(X)
        self.Y = np.array(Y).astype(int)
        if self.X.ndim != 2 or self.X.shape[0] != self.Y.shape[0] or self.Y.ndim != 1:
            raise ValueError("invalid input")

    def predict(self, X, k=5):
        dists = np.linalg.norm(X[:, np.newaxis, :] - self.X, axis=-1)
        idx = np.argsort(dists, axis=-1)[:, :k]
        pred = [Counter(self.Y[i]).most_common(1)[0][0] for i in idx]
        return np.asarray(pred)

if __name__ == "__main__":
    cls1 = KnearestNeighbors()
    cls2 = KNeighborsClassifier()
    evaluate_classifier(cls1, cls2)