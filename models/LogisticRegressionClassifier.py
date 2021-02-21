import tensorflow as tf
import numpy as np

class BinaryLogisticRegression:
    def __init__(self, max_iter = 30, learning_rate = 0.01):
        self.input_dim = None
        self.max_iter = max_iter
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        return 1.0 / (1.0 + tf.exp(-(tf.matmul(x, self.w) + self.b)))

    def safeLog(self, y):
        return tf.math.log(tf.clip_by_value(y, 1e-10, 1.0))

    def fit(self, x, y):
        self.input_dim = x.shape[1]
        input_sample = tf.Variable(x, dtype=tf.float32)
        input_label = tf.Variable(y, dtype=tf.float32)
        input_label = tf.reshape(input_label, shape=(-1, 1))
        self.optimizer = tf.optimizers.Adam(learning_rate=self.learning_rate)
        self.w = tf.Variable(tf.random.truncated_normal(shape=(self.input_dim, 1)))
        self.b = tf.Variable(tf.ones(shape=1), dtype=tf.float32)
        for epoch in range(self.max_iter):
            with tf.GradientTape() as tape:
                predict = self.sigmoid(input_sample)
                loss = -(1.0 - input_label) * self.safeLog(1 - predict) - input_label * self.safeLog(predict)
                loss = tf.reduce_mean(loss)
                grads = tape.gradient(loss, [self.w, self.b])
            self.optimizer.apply_gradients(zip(grads, [self.w, self.b]))

    def predict(self, x):
        return self.sigmoid(tf.convert_to_tensor(x, dtype=tf.float32)).numpy()

    def score(self, x, y):
        prediction = self.predict(x) >= 0.5
        prediction = prediction[:, 0] == y
        acc = np.sum(prediction) * 1.0 / y.shape[0]
        return acc