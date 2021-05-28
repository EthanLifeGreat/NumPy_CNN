import numpy as np


def encode_one_hot(y):
    _num_samples = len(y)
    _num_classes = np.max(y) + 1
    _y_01 = np.zeros([_num_samples, _num_classes])
    _y_01[np.arange(0, y.shape[0]), y.ravel()] = 1
    return _y_01
