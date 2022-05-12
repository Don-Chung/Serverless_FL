import numpy as np

train_seed = 50
data = {}


def LocalTrain(r, c, train_seed):
    np.random.seed(train_seed)
    return np.random.rand(r, c)
