import numpy as np
from sklearn.datasets import make_blobs
from sklearn import cluster
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_idx = None
        self.cluster_centers = None

    def __clustering__(self, X):
        dist_sum = 0
        for idx, x in enumerate(X):
            dists = np.linalg.norm(x - self.cluster_centers, axis=-1)
            self.cluster_idx[idx] = np.argmin(dists)
            dist_sum += np.min(dists)
        for cluster_idx in range(self.n_clusters):
            self.cluster_centers[cluster_idx] = np.mean(X[self.cluster_idx == cluster_idx], axis=0)
        return dist_sum
    def fit(self, X):
        self.cluster_idx = np.zeros((X.shape[0]), dtype=int)
        self.cluster_centers = X[np.random.choice(np.arange(X.shape[0]), self.n_clusters, replace=False)]
        old_dist_sum = self.__clustering__(X)
        while True:
            dist_sum = self.__clustering__(X)
            if old_dist_sum != dist_sum:
                old_dist_sum = dist_sum
            else:
                break
        for cluster_idx in range(self.n_clusters):
            idx = self.cluster_idx == cluster_idx
            plt.scatter(X[idx, 0], X[idx, 1])
        plt.show()


if __name__ == "__main__":
    n_samples = 100
    n_features = 2
    centers = 3
    blobs = make_blobs(n_samples=n_samples, n_features=n_features, centers=centers)
    kmeans = KMeans(centers)
    kmeans.fit(blobs[0])


