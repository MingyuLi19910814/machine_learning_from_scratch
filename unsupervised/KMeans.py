import numpy as np
from sklearn.datasets import make_blobs
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self):
        self.k = None
        self.centers = None
        self.cluster_idx = None

    def __clustering__(self, X):
        dist_square_sum = 0
        for idx, x in enumerate(X):
            dists = np.sum((x - self.centers) ** 2, axis=-1)
            self.cluster_idx[idx] = np.argmin(dists)
            dist_square_sum += dists[self.cluster_idx[idx]]
        for cluster_idx in range(self.k):
            self.centers[cluster_idx] = np.mean(X[self.cluster_idx == cluster_idx], axis=0)
        return dist_square_sum

    def fit(self, X, k):
        X = np.asarray(X)
        if X.shape[0] == 0:
            raise ValueError("invalid input")
        self.cluster_idx = np.zeros(shape=(X.shape[0]), dtype=int)
        assert k <= X.shape[0]
        self.k = k
        self.centers = X[np.random.choice(np.arange(X.shape[0]), self.k, replace=False)]
        old_dist_square_sum = self.__clustering__(X)
        while True:
            dist_square_sum = self.__clustering__(X)
            if old_dist_square_sum != dist_square_sum:
                old_dist_square_sum = dist_square_sum
            else:
                break

if __name__ == "__main__":
    n_samples = 100
    n_features = 2
    n_centers = 4

    X, Y = make_blobs(n_samples=n_samples,
                      n_features=n_features,
                      centers=n_centers,
                      cluster_std=0.3)
    cluster = KMeans()
    cluster.fit(X, n_centers)
    for cluster_idx in range(n_centers):
        pts = X[cluster.cluster_idx==cluster_idx]
        plt.scatter(pts[:, 0], pts[:, 1])
    plt.show()
