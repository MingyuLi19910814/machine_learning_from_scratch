import numpy as np
from sklearn import preprocessing, model_selection, datasets, metrics
from matplotlib import pyplot as plt

class Dense:
    def __init__(self, input_shape, output_shape, lr):
        self.lr = lr
        self.w = np.random.randn(input_shape, output_shape) * 0.01
        self.b = np.random.randn(output_shape)
        self.X = None
    def forward(self, X):
        self.X = X
        return np.dot(X, self.w) + self.b
    def backward(self, dout):
        dw = np.dot(self.X.T, dout)
        db = np.mean(dout, axis=0)
        dout = np.dot(dout, self.w.T)
        self.w -= self.lr * dw
        self.b -= self.lr * db
        return dout

class ReLU:
    def __init__(self):
        self.mask = None
    def forward(self, X):
        self.mask = X < 0
        out = np.copy(X)
        out[self.mask] = 0
        return out
    def backward(self, dout):
        out = np.copy(dout)
        out[self.mask] = 0
        return out

class SoftmaxCrossEntropy:
    def __init__(self):
        pass
    def forward(self, X):
        C = np.max(X)
        exp = np.exp(X - C)
        return exp / np.sum(exp, axis=-1, keepdims=True)
    def backward(self, gt, pred):
        return (pred - gt) / gt.shape[0]
    def loss(self, gt, pred):
        return -np.mean(gt * np.log(pred + 1e-13))

class NeuralNetwork:
    def __init__(self, shapes, lr=0.01):
        self.layers = []
        for i in range(shapes.__len__() - 1):
            self.layers.append(Dense(shapes[i], shapes[i + 1], lr))
            self.layers.append(ReLU())
        self.layers.pop()
        self.last_layer = SoftmaxCrossEntropy()

    def fit(self, X, gt, max_it):
        losses = []
        for it in range(max_it):
            x = np.copy(X)
            for layer in self.layers:
                x = layer.forward(x)
            pred_y = self.last_layer.forward(x)
            losses.append(self.last_layer.loss(gt, pred_y))
            dout = self.last_layer.backward(gt, pred_y)
            for layer in self.layers[::-1]:
                dout = layer.backward(dout)
        #print(losses)
        plt.plot(losses)
        plt.show()

    def predict(self, X):
        x = np.copy(X)
        for layer in self.layers:
            x = layer.forward(x)
        return np.argmax(x, axis=-1)

if __name__ == "__main__":
    # iris = datasets.load_iris()
    # X = iris.data
    # Y = iris.target
    # one_hot_encoder = preprocessing.LabelBinarizer()
    # Y = one_hot_encoder.fit_transform(Y)
    # train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.2)
    # nn = NeuralNetwork(shapes=(X.shape[1], 8, one_hot_encoder.classes_.__len__()))
    # nn.fit(train_x, train_y, 5000)
    # pred_y = nn.predict(test_x)
    # test_y = one_hot_encoder.inverse_transform(test_y)
    # accuracy = metrics.accuracy_score(test_y, pred_y)
    # print(accuracy)
    n_samples = 1000
    n_features = 20
    n_informative = 15
    n_redundant = 0
    n_classes = 3
    X, Y = datasets.make_classification(n_samples=n_samples,
                                        n_classes=n_classes,
                                        n_features=n_features,
                                        n_informative=n_informative,
                                        n_redundant=n_redundant,
                                        random_state=0)
    scaler = preprocessing.StandardScaler()
    X = scaler.fit_transform(X)
    one_hot_encoder = preprocessing.LabelBinarizer()
    Y = one_hot_encoder.fit_transform(Y)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=0)
    nn = NeuralNetwork(shapes=(n_features, 8, 12, n_classes), lr=0.1)
    nn.fit(train_x, train_y, max_it=10000)
    pred_y = nn.predict(test_x)
    test_y = one_hot_encoder.inverse_transform(test_y)
    acc = metrics.accuracy_score(test_y, pred_y)
    print(pred_y, test_y)
    print('acc = {}'.format(acc))