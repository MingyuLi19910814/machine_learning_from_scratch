import numpy as np
from matplotlib import pyplot as plt

class KMeansClustering:
    def __init__(self, k, points):
        self.k = k
        self.points = np.array(points)
        if self.points.ndim == 1:
            self.points = np.reshape(self.points, (self.points.__len__(), 1))
        self.__clustering__()

    def getClusteredPointIdx(self):
        res = []
        for _ in range(self.k):
            res.append(np.where(self.nearestClusterIdx == _))
        return res

    def __getNearestCenter__(self, centers):
        # self.points: (n, m)
        # centers: (k, m)
        dist2centers = np.linalg.norm(self.points[:, np.newaxis, :] - centers, axis=-1)
        nearestClusterIdx = np.argmin(dist2centers, axis=-1)
        distSum = np.sum(np.min(dist2centers, axis=-1))
        return nearestClusterIdx, distSum

    def __clustering__(self):
        centers = self.points[np.random.choice(self.points.__len__(), self.k, replace=False)]
        lastDistSum = np.inf
        epsilon = 1e-8
        nearestClusterIdx, distSum = self.__getNearestCenter__(centers)
        while np.abs(distSum - lastDistSum) > epsilon:
            for clusterIdx in range(self.k):
                pointIdx = np.where(nearestClusterIdx == clusterIdx)
                centers[clusterIdx] = np.average(self.points[pointIdx], axis=0)
            lastDistSum = distSum
            nearestClusterIdx, distSum = self.__getNearestCenter__(centers)
        self.nearestClusterIdx = nearestClusterIdx

if __name__ == "__main__":
    x1 = np.random.rand(10, 2) + 5
    x2 = np.random.rand(10, 2) - 5
    X = np.concatenate([x1, x2], axis=0)
    k = 2
    clusters = KMeansClustering(k, X).getClusteredPointIdx()
    for cluster in clusters:
        plt.scatter(X[cluster, 0], X[cluster, 1])
    plt.show()

