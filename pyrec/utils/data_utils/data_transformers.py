import scipy.sparse
import numpy as np


class NoTransform(object):
    """docstring for NoTransform"""

    def __init__(self):
        super(NoTransform, self).__init__()

    def fit_transform(self, data):
        return data

    def update(self, params):
        pass

