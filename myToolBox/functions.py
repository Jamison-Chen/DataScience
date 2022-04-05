import abc
import numpy as np
from copy import deepcopy


class ActivationFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.forwardInput = None
        self.forwardOutput = None

    @abc.abstractmethod
    def forward(self, *args):
        return NotImplemented

    @abc.abstractmethod
    def backward(self):
        return NotImplemented


class LossFunction(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __init__(self):
        self.y = None
        self.y_pred = None
        self.forwardOutput = None

    @abc.abstractmethod
    def forward(self, *args):
        return NotImplemented

    @abc.abstractmethod
    def backward(self):
        return NotImplemented


class Sigmoid(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.forwardInput = deepcopy(z)
        self.forwardOutput = 1 / (1 + np.exp(-z))
        return self.forwardOutput

    def backward(self):
        return self.forwardOutput * (1 - self.forwardOutput)


class Softmax(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.forwardInput = deepcopy(z)
        e = np.exp(z)
        self.forwardOutput = e / (np.sum(e, axis=1)[:, np.newaxis])
        return self.forwardOutput

    def backward(self):
        return self.forwardOutput * (1 - self.forwardOutput)


class ReLU(ActivationFunction):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        self.forwardInput = deepcopy(z)
        self.forwardOutput = np.where(z <= 0, 0, z)
        return self.forwardOutput

    def backward(self):
        return np.where(self.forwardInput <= 0, 0, 1)


# class Maxout(ActivationFunction):


class CrossEntropy(LossFunction):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_pred):
        self.y = deepcopy(y)
        self.y_pred = deepcopy(y_pred)
        self.forwardOutput = np.sum(self.y * np.log(self.y_pred)) * -1
        return self.forwardOutput

    def backward(self):
        return ((1 - self.y) / (1 - self.y_pred)) - (self.y / self.y_pred)


class SquaredError(LossFunction):
    def __init__(self):
        super().__init__()

    def forward(self, y, y_pred):
        self.y = deepcopy(y)
        self.y_pred = deepcopy(y_pred)
        self.forwardOutput = np.sum(np.square(self.y - self.y_pred)) * 0.5
        return self.forwardOutput

    def backward(self):
        return self.y_pred - self.y
