import numpy as np
import scipy.sparse
from sklearn.cross_validation import train_test_split


def generateTrainValidationRandom(mat, percent=10, seed=None):
    if scipy.sparse.isspmatrix(mat):
        total_items = len(mat.data)
        num_validation_items = int(total_items * percent / 100)
        indices = np.random.choice(
            np.arange(len(mat.data)), num_validation_items, replace=False)
        train = mat.copy()
        validation = mat.copy()
        train.data[indices] = 0.0
        validation.data[~np.in1d(np.arange(total_items), indices)] = 0.0
        train.eliminate_zeros()
        validation.eliminate_zeros()
    else:
        train, validation = train_test_split(mat, test_size=(percent / 100.0))
    return train, validation