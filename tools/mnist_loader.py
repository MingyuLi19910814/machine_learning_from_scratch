import pandas
import os
from pathlib import Path
import numpy as np
from sklearn.model_selection import train_test_split

class DataProcessor:
    def __init__(self, train='train', interval = 128):

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


DataProcessor()