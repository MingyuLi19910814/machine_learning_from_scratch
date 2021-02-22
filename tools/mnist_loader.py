import pandas
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, train='train', interval=128, batch_size=1024):
        self.idx = 0
        self.batch_size = batch_size
        current_path = os.path.abspath(__file__)
        train_path = Path(current_path).parent.parent / 'data/mnist_train.csv'
        test_path = Path(current_path).parent.parent / 'data/mnist_test.csv'
        assert os.path.isfile(train_path) and os.path.isfile(test_path)

        fd = pandas.read_csv(train_path if train == 'train' else test_path).to_numpy()
        self.x = fd[:, 1:]
        self.y = fd[:, 0]
        self.x = self.x // interval


    def get(self, val_size=0.0):
        if val_size > 0:
            train_x, val_x, train_y, val_y = train_test_split(self.x, self.y, random_state=42, test_size=val_size)
            return train_x, train_y, val_x, val_y
        else:
            return self.x, self.y

    def getNextBatch(self):
        if self.idx + self.batch_size <= self.x.shape[0]:
            batch_x = self.x[self.idx:self.idx + self.batch_size, :]
            batch_y = self.y[self.idx:self.idx + self.batch_size]
            self.idx += self.batch_size
            return batch_x, batch_y, True
        else:
            batch_x = self.x[self.idx:, :]
            batch_y = self.y[self.idx:]
            self.idx = 0
            return batch_x, batch_y, False