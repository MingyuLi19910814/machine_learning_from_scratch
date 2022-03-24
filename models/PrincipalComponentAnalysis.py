import numpy as np
from sklearn.datasets import load_iris
from sklearn import decomposition
class PrincipalComponentAnalysis:
    def __init__(self):
        pass

    def fit(self, X, n_component):
        x_std = X - np.mean(X, axis=0)
        cov_mat = np.cov(x_std, rowvar=False)
        eig_vals, eig_vecs = np.linalg.eigh(cov_mat)
        sorted_idx = np.argsort(eig_vals)[::-1]
        self.n_component = n_component
        self.eig_vecs = eig_vecs[:, sorted_idx]
        self.eig_vals = eig_vals[sorted_idx]
        return np.dot(x_std, self.eig_vecs[:, :self.n_component])

    def getExplainedVariantRatio(self):
        return self.eig_vals[:self.n_component] / np.sum(self.eig_vals)

if __name__ == "__main__":
    n_component = 3
    n_sample = -1
    iris = load_iris()
    X = iris.data[:n_sample,:]
    Y = iris.target

    PCA = PrincipalComponentAnalysis()
    x1 = np.abs(PCA.fit(X, n_component))

    skPCA = decomposition.PCA(n_components=n_component)
    x2 = np.abs(skPCA.fit_transform(X))
    error = np.linalg.norm(x1 - x2)
    print(error)
