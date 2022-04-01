import numpy as np


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    e = np.exp(z)
    return e / (np.sum(e, axis=1)[:, np.newaxis])


def crossEntropy(y, y_pred):
    return np.sum(y * np.log(y_pred)) * -1


def MSE(y, y_pred):
    return np.sum(np.square(y - y_pred)) * 0.5
