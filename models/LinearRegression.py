import numpy as np

class LinearRegression:
    def __init__(self, X, Y):
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        if self.X.shape[0] != self.Y.shape[0]:
            raise ValueError("inconsistency between X and Y sample number!")
        elif self.X.shape[0] == 0:
            raise ValueError("input sample can not be empty!")
        elif self.Y.ndim != 1:
            raise ValueError("Y must be one dimensional")
        self.__predict__()

    def getSlope(self):
        return self.slope

    def getIntersect(self):
        return self.intersection

    def __predict__(self):
        meanX = np.mean(self.X, axis=0)
        meanY = np.mean(self.Y)
        self.slope = np.sum((self.X - meanX) * (self.Y - meanY)) / np.sum((self.X - meanX) ** 2)
        self.intersection = meanY - self.slope * meanX

if __name__ == "__main__":
    slope = 3
    intersect = -30
    X = np.arange(0, 100)
    Y = slope * X + intersect + np.random.rand(*X.shape)
    regression = LinearRegression(X, Y)
    print(regression.getSlope())
    print(regression.getIntersect())