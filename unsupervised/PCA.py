import numpy as np
from sklearn import decomposition
from sklearn.datasets import load_iris


class PCA:
    def __init__(self):
        pass

    def fit_transform(self, X, k):
        x_center = X - np.mean(X, axis=0)
        x_cov = np.cov(x_center, rowvar=False)
        eigen_values, eigen_vectors = np.linalg.eigh(x_cov)
        idx = np.argsort(eigen_values)[::-1][:k]
        self.eigen_values = eigen_values[idx]
        self.eigen_vectors = eigen_vectors[:, idx]
        self.explained_variance_ = np.cumsum(self.eigen_values)
        return np.dot(x_center, self.eigen_vectors)

if __name__ == "__main__":
    X = load_iris().data
    n_components = 3
    reduced_x1 = PCA().fit_transform(X, n_components)
    reduced_x2 = decomposition.PCA(n_components=n_components).fit_transform(X)
    for idx in range(reduced_x1.shape[1]):
        err1 = np.linalg.norm(reduced_x1[:, idx] - reduced_x2[:, idx])
        err2 = np.linalg.norm(reduced_x1[:, idx] + reduced_x2[:, idx])
        err = min(err1, err2)
        print('dim = {}, err = {}'.format(idx, err))
