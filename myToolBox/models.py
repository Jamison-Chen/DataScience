import abc
import random
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
from .functions import (
    ActivationFunction,
    LossFunction,
    Sigmoid,
    Softmax,
    SquaredError,
    CrossEntropy,
)


class Model(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def fit(self, x, y):
        return NotImplemented

    def predict(self, x):
        return NotImplemented


class KNN(Model):
    def __init__(self, k):
        self.__k = k
        self.__x = None
        self.__y = None

    def fit(self, x, y):
        self.__x = deepcopy(x)
        self.__y = deepcopy(y)
        # Time Complexity: O(n*m)

    def predict(self, x):
        result = []
        for each in x:
            # Use "Euclidean Distance"
            distances = np.linalg.norm(each - self.__x, axis=1)
            indices = np.argpartition(distances, self.__k)[: self.__k]
            uniqueVal, counts = np.unique(
                np.squeeze(self.__y[indices, :]), return_counts=True
            )
            result.append(uniqueVal[counts.argmax()])
        # Time Complexity: O(n*n*m)
        return np.array([result]).T


class GenerativeModel(Model):
    def __init__(self, share_cov=False, naive_bayes=False):
        self.__mu = {}
        self.__cov = {}
        self.__p = {}
        self.__shareCov = share_cov
        self.__meanCov = None
        self.__naiveBayes = naive_bayes

    def fit(self, x, y):
        n, m = x.shape
        uniqueVals, counts = np.unique(y, return_counts=True)
        for i, classIdx in enumerate(uniqueVals):
            groupSamples = x[np.where(y.squeeze() == classIdx)[0], :]
            self.__mu[classIdx] = np.mean(groupSamples, axis=0)

            # Time complexity of claculating covariances: O(n*m)
            self.__cov[classIdx] = np.cov(groupSamples.T)
            if self.__naiveBayes:
                # Assume covariance between different features to be 0.
                self.__cov[classIdx] *= np.eye(2)
            self.__p[classIdx] = counts[i] / n
        if self.__shareCov:
            self.__meanCov = np.zeros((m, m))
            for classIdx in self.__cov:
                self.__meanCov += self.__p[classIdx] * self.__cov[classIdx]
        # Time Complexity: O(c*n*m), where c denotes the number of classes.

    def predict(self, x):
        n = x.shape[0]
        pMax = np.zeros((n, 1))
        prediction = np.empty((n, 1), dtype=np.int8)
        for c in self.__mu:
            # Assume Normal (Gaussian) Distribution
            # `Pxlc` denotes P(x|c)
            Pxlc = multivariate_normal.pdf(
                x,
                mean=self.__mu[c],
                cov=self.__meanCov if self.__shareCov else self.__cov[c],
            )
            p = (self.__p[c] * Pxlc).reshape(n, 1)
            prediction = np.where(p > pMax, c, prediction)
            pMax = np.maximum(p, pMax)
        # Time Complexity: O(c*n*m)
        return prediction


class DichotomousLogistic(Model):
    def __init__(self):
        self.__w = None
        self.__b = 0
        self.__classes = []
        self.sigmoid = Sigmoid()
        self.epoch = None
        self.lossHistory = []

    def __crossEntropy(self, y, y_pred):
        return -1 * np.sum(
            np.where(y == self.__classes[0], np.log(y_pred), np.log(1 - y_pred))
        )

    def fit(self, x, y, lr, epoch):
        self.__w = np.random.uniform(-1, 1, (x.shape[1],))
        self.__b = np.random.uniform(-1, 1)
        self.__classes = np.unique(y)
        self.lossHistory = []
        self.epoch = epoch
        y_squeezed = deepcopy(y).squeeze()
        y_hat = np.where(y_squeezed == self.__classes[0], 1, 0)

        for _ in range(epoch):
            y_pred = self.sigmoid.forward(x @ self.__w + self.__b)

            # Store loss (for ploting)
            self.lossHistory.append(
                self.__crossEntropy(y_squeezed, y_pred) / y_pred.shape[0]
            )

            # update w and b using gradient descent
            self.__w -= np.sum((y_pred - y_hat)[:, np.newaxis] * x, axis=0) * lr
            self.__b -= np.sum((y_pred - y_hat)[:, np.newaxis], axis=0) * lr

    def predict(self, x):
        y_pred = self.sigmoid.forward(x @ self.__w + self.__b)
        return np.where(y_pred >= 0.5, self.__classes[0], self.__classes[1])[
            :, np.newaxis
        ]


class LogisticRegression(Model):
    def __init__(self):
        self.__w = None
        self.__b = None
        self.__classes = {}
        self.softmax = Softmax()
        self.crossEntropy = CrossEntropy()
        self.epoch = None
        self.lossHistory = []

    def fit(self, x_train, y_train, lr=0.005, epoch=200):
        uniqVals = np.unique(y_train)
        self.__w = np.random.uniform(-1, 1, (x_train.shape[1], len(uniqVals)))
        self.__b = np.random.uniform(-1, 1, (len(uniqVals),))
        self.lossHistory = []
        self.epoch = epoch
        y = deepcopy(y_train)

        # Encode class names into numerical numbers
        for i, each in enumerate(uniqVals):
            y[y == each] = i
            self.__classes[i] = each

        # Transform the true answers into vectors
        y = np.where(np.arange(len(uniqVals)) == y, 1, 0)

        # Training
        for _ in range(epoch):
            y_pred = self.softmax.forward(x_train @ self.__w + self.__b)

            # Store loss (for ploting)
            self.lossHistory.append(
                self.crossEntropy.forward(y, y_pred) / y_pred.shape[0]
            )

            # Directly derive the final gradient after partial differential
            # w_gradient = x_train.T @ (y_pred - y)
            # b_gradient = np.sum((y_pred - y), axis=0)

            # Chain the backpropagations to derive the gradient
            w_gradient = x_train.T @ (
                self.softmax.backward() * self.crossEntropy.backward()
            )
            b_gradient = np.sum(
                (self.softmax.backward() * self.crossEntropy.backward()), axis=0
            )

            # Update w and b using gradient descent
            self.__w -= w_gradient * lr
            self.__b -= b_gradient * lr

    def predict(self, x):
        y_pred = self.softmax.forward(x @ self.__w + self.__b)

        # Decode numerical numbers to class names
        ans = [self.__classes[each] for each in np.argmax(y_pred, axis=1)]

        return np.array(ans)[: np.newaxis]


class TL_FC_NN(Model):
    def __init__(self, hiddenLayerNodeNumber):
        self.__w1 = None
        self.__w2 = None
        self.__classes = {}
        self.__numOfNode = hiddenLayerNodeNumber
        self.epoch = None
        self.lossHistory = []

        # `h` denotes the activation function of the hidden layer
        # defaults to sigmoid
        self.__h = Sigmoid()

        # `l` denotes the loss function
        # defaults to square error
        self.__l = SquaredError()

        # softmax for cross entropy loss (if was choosen)
        self.softmax = Softmax()

    @property
    def activation_function(self):
        return self.__h

    @activation_function.setter
    def activation_function(self, f):
        if isinstance(f, ActivationFunction):
            self.__h = f
        else:
            raise Exception(
                "Please provide an activation function instead of {}".format(type(f))
            )

    @property
    def loss_function(self):
        return self.__l

    @loss_function.setter
    def loss_function(self, l):
        if isinstance(l, LossFunction):
            self.__l = l
        else:
            raise Exception(
                "Please provide an activation function instead of {}".format(type(l))
            )

    def fit(self, x_train, y_train, lr, batch_size=None, epoch=1000):
        # `batch_size` defaults to full-batch
        if not batch_size:
            batch_size = x_train.shape[0]

        # Initialize weights
        uniqVals = np.unique(y_train)
        self.__w1 = np.random.uniform(-1, 1, (x_train.shape[1], self.__numOfNode))
        self.__w2 = np.random.uniform(-1, 1, (self.__numOfNode, len(uniqVals)))
        self.lossHistory = []
        self.epoch = epoch
        x = deepcopy(x_train)
        y = deepcopy(y_train)

        # Encode each class name to numerical numbers
        for i, each in enumerate(uniqVals):
            y[y == each] = i
            self.__classes[i] = each

        # Transform the true answers into vectors
        y = np.where(np.arange(len(uniqVals)) == y, 1, 0)

        # Slice the batches
        batches = []
        temp = list(range(x_train.shape[0]))
        random.shuffle(temp)
        while len(temp) > 0:
            argsliced = temp[:batch_size]
            batches.append({"x": x[argsliced, :], "y": y[argsliced, :]})
            temp = temp[batch_size:]

        # Training
        for _ in range(epoch):
            batchLoss = 0
            for batch in batches:
                # Forward
                h = self.__h.forward(batch["x"] @ self.__w1)  # (n, k)
                if isinstance(self.__l, CrossEntropy):
                    y_pred = self.softmax.forward(h @ self.__w2)  # (n, c)
                else:
                    y_pred = h @ self.__w2  # (n, c)

                # Accumulate batch loss
                batchLoss += self.__l.forward(batch["y"], y_pred) / y_pred.shape[0]

                # Backward
                if isinstance(self.__l, CrossEntropy):
                    w2_gradient = h.T @ (self.softmax.backward() * self.__l.backward())
                else:
                    w2_gradient = h.T @ self.__l.backward()  # (k, c)

                w1_gradient = batch["x"].T @ (
                    self.__h.backward() * (self.__l.backward() @ self.__w2.T)
                )  # (m, k)

                self.__w2 -= w2_gradient * lr  # (k, c)
                self.__w1 -= w1_gradient * lr  # (m, k)
            # Store loss (for ploting)
            self.lossHistory.append(batchLoss)

    def predict(self, x):
        h = self.__h.forward(x @ self.__w1)  # (n, k)
        if isinstance(self.__l, CrossEntropy):
            y_pred = self.softmax.forward(h @ self.__w2)  # (n, c)
        else:
            y_pred = h @ self.__w2  # (n, c)
        ans = [self.__classes[each] for each in np.argmax(y_pred, axis=1)]
        return np.array(ans)[: np.newaxis]


if __name__ == "__main__":
    x = np.random.rand(12, 2)
    pre = np.random.rand(12, 1)
    y = np.where(
        pre >= 2 / 3,
        2,
        np.where(pre >= 1 / 3, 1, 0),
    )
    tlnn = TL_FC_NN(10, 0.00001, 8000)
    tlnn.fit(x, y)
    tlnn.predict(x)
