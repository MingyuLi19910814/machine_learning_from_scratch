import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
class Dense:
    def __init__(self, input_shape, output_shape, lr):
        self.lr = lr
        self.w = np.random.randn(input_shape, output_shape) * 0.01
        self.b = np.random.randn(output_shape)

    def forward(self, X):
        self.X = X
        pred = np.dot(X, self.w) + self.b
        return pred

    def backward(self, dout):
        dw = np.dot(self.X.T, dout)
        db = np.mean(dout, axis=0)
        dout = np.dot(dout, self.w.T)
        self.w = self.w - self.lr * dw
        self.b = self.b - self.lr * db
        return dout

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, X):
        self.mask = X < 0
        X[self.mask] = 0
        return X

    def backward(self, dout):
        dout[self.mask] = 0
        return dout

class SoftmaxCrossEntropy:
    def __init__(self):
        pass

    def forward(self, X):
        C = np.max(X)
        exp = np.exp(X - C)
        return exp / np.sum(exp, axis=-1, keepdims=True)

    def backward(self, pred, gt):
        return (pred - gt) / gt.shape[0]

class NeuralNetwork:
    def __init__(self, shapes, lr, max_it):
        self.layers = []
        for i in range(shapes.__len__() - 1):
            self.layers.append(Dense(shapes[i], shapes[i + 1], lr))
            self.layers.append(ReLU())
        self.layers.pop()
        self.last_layer = SoftmaxCrossEntropy()
        self.max_it = max_it
        self.encoder = LabelBinarizer()
    def fit(self, X, gt):
        X = np.asarray(X)
        gt = np.asarray(gt)
        gt = self.encoder.fit_transform(gt)
        for it in range(self.max_it):
            x = np.copy(X)
            for layer in self.layers:
                x = layer.forward(x)
            pred = self.last_layer.forward(x)
            dout = self.last_layer.backward(pred, gt)
            for layer in self.layers[::-1]:
                dout = layer.backward(dout)

    def predict(self, X):
        x = np.copy(X)
        for layer in self.layers:
            x = layer.forward(x)
        pred = np.argmax(x, axis=-1)
        return pred

if __name__ == "__main__":
    n_samples = 1000
    n_classes = 3
    n_features = 10
    n_informative = 8
    X, Y = make_classification(n_samples=n_samples,
                               n_classes=n_classes,
                               n_features=n_features,
                               n_informative=n_informative,
                               random_state=0)
    # X = StandardScaler().fit_transform(X)
    train_x, test_x, train_y, test_y = train_test_split(X, Y, test_size=0.2, random_state=0)
    cls = NeuralNetwork([train_x.shape[1], 8, n_classes], lr=0.1, max_it=1000)
    cls.fit(train_x, train_y)
    pred_y = cls.predict(test_x)
    acc = accuracy_score(test_y, pred_y)
    print('acc = {}'.format(acc))