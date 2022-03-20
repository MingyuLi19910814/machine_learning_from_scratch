import numpy as np
from collections import OrderedDict
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class Dense:
    def __init__(self, input_shape, output_shape, lr):
        self.W = 0.01 * np.random.randn(input_shape, output_shape)
        self.b = np.random.randn(output_shape)
        self.dW = np.zeros_like(self.W)
        self.db = None
        self.X = None
        self.lr = lr
    def forward(self, X):
        self.X = X.copy()
        return np.dot(X, self.W) + self.b
    def backward(self, dout):
        self.dW = np.dot(self.X.T, dout)
        self.db = np.sum(dout, axis=0)
        self.W -= self.lr * self.dW
        self.b -= self.lr * self.db
        out = np.dot(dout, self.W.T)
        #print('dW = {}, X = {}, dout={}'.format(self.dW[0], self.X.T[0], dout))
        #print('Dense backward = {}'.format(out))
        return out

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X < 0
        out = X.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        out = dout.copy()
        out[self.mask] = 0
        #print('ReLU backward = {}'.format(out))
        return out


class SoftmaxCrossEntropy:
    def __init__(self):
        self.out = None

    def forward(self, x):
        C = np.max(x)
        exp = np.exp(x - C)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def loss(self, y, gt):
        out = np.sum(-gt * np.log(y + 1e-13)) / gt.shape[0]
        #print('loss = {}'.format(out))
        return out

    def backward(self, y, gt):
        out = (y - gt) / y.shape[0]
        #print('softmax backward = {}'.format(out))
        return out



class TwoLayerNet:
    def __init__(self, input_shape, hidden_shape, output_shape, lr):
        self.layers = []
        self.layers.append(Dense(input_shape, hidden_shape, lr))
        self.layers.append(ReLU())
        self.layers.append(Dense(hidden_shape, output_shape, lr))
        self.lastLayer = SoftmaxCrossEntropy()

    def train(self, input, gt, max_it):
        loss = []
        for it in range(max_it):
            x = input
            for layer in self.layers:
                x = layer.forward(x)
            y = self.lastLayer.forward(x)
            loss.append(self.lastLayer.loss(y, gt))
            dout = self.lastLayer.backward(y, gt)
            for i in range(self.layers.__len__())[::-1]:
                dout = self.layers[i].backward(dout)
        print(loss)

    def predict(self, x):
        x = np.copy(x)
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=-1)

def one_hot(Y, num_class):
    gt = np.zeros(shape=(Y.shape[0], num_class), dtype=float)
    gt[np.arange(Y.shape[0]), Y.astype(int)] = 1
    return gt

if __name__ == "__main__":
    iris = load_iris()
    X = iris.data
    Y = iris.target
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.25)
    train_x /= np.max(X)
    test_x /= np.max(X)
    train_y = one_hot(train_y, 3)
    network = TwoLayerNet(input_shape=X.shape[1], hidden_shape=8, output_shape=np.max(Y) + 1, lr=0.3)
    network.train(train_x, train_y, max_it=500)
    predict = network.predict(test_x)
    accuracy = accuracy_score(test_y, predict)
    print(accuracy)