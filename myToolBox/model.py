import abc
import numpy as np
from scipy.stats import multivariate_normal
from copy import deepcopy
from .my_math import sigmoid, softmax, MSE, crossEntropy


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
    def __init__(self, lr, numOfLoop):
        self.__w = None
        self.__b = 0
        self.__classes = []
        self.__lr = lr
        self.numOfLoop = numOfLoop
        self.loss = []

    def __crossEntropy(self, y, y_pred):
        return -1 * np.sum(
            np.where(y == self.__classes[0], np.log(y_pred), np.log(1 - y_pred))
        )

    def fit(self, x, y):
        self.__w = np.random.uniform(-1, 1, (x.shape[1],))
        self.__b = np.random.uniform(-1, 1)
        self.__classes = np.unique(y)
        self.loss = []
        y_squeezed = deepcopy(y).squeeze()
        y_hat = np.where(y_squeezed == self.__classes[0], 1, 0)

        for _ in range(self.numOfLoop):
            y_pred = sigmoid(x @ self.__w + self.__b)

            # Store loss (for ploting)
            self.loss.append(self.__crossEntropy(y_squeezed, y_pred))

            # update w and b using gradient descent
            self.__w -= np.sum((y_pred - y_hat)[:, np.newaxis] * x, axis=0) * self.__lr
            self.__b -= np.sum((y_pred - y_hat)[:, np.newaxis], axis=0) * self.__lr

    def predict(self, x):
        y_pred = sigmoid(x @ self.__w + self.__b)
        return np.where(y_pred >= 0.5, self.__classes[0], self.__classes[1])[
            :, np.newaxis
        ]


class LogisticRegression(Model):
    def __init__(self, lr=0.005, numOfLoop=200):
        self.__w = None
        self.__b = None
        self.__classes = {}
        self.__lr = lr
        self.numOfLoop = numOfLoop
        self.loss = []

    def fit(self, x_train, y_train):
        uniqVals = np.unique(y_train)
        self.__w = np.random.uniform(-1, 1, (x_train.shape[1], len(uniqVals)))
        self.__b = np.random.uniform(-1, 1, (len(uniqVals),))
        self.loss = []
        y = deepcopy(y_train)

        # Encode class names into numerical numbers
        for i, each in enumerate(uniqVals):
            y[y == each] = i
            self.__classes[i] = each

        # Transform the true answers into vectors
        y = np.where(np.arange(len(uniqVals)) == y, 1, 0)

        # Training
        for _ in range(self.numOfLoop):
            y_pred = softmax(x_train @ self.__w + self.__b)

            # Store loss (for ploting)
            self.loss.append(crossEntropy(y, y_pred))

            # update w and b using gradient descent
            self.__w -= x_train.T @ (y_pred - y) * self.__lr
            self.__b -= np.sum((y_pred - y), axis=0) * self.__lr

    def predict(self, x):
        y_pred = softmax(x @ self.__w + self.__b)

        # Decode numerical numbers to class names
        ans = [self.__classes[each] for each in np.argmax(y_pred, axis=1)]

        return np.array(ans)[: np.newaxis]


class TwoLayerNN(Model):
    def __init__(self, numOfNode, lr, numOfLoop):
        self.__w1 = None
        self.__w2 = None
        self.__classes = {}
        self.__numOfNode = numOfNode
        self.__lr = lr
        self.numOfLoop = numOfLoop
        self.loss = []

    def fit(self, x_train, y_train):
        uniqVals = np.unique(y_train)
        self.__w1 = np.random.uniform(-1, 1, (x_train.shape[1], self.__numOfNode))
        self.__w2 = np.random.uniform(-1, 1, (self.__numOfNode, len(uniqVals)))
        self.loss = []
        y = deepcopy(y_train)

        # Encode each class name to numerical numbers
        for i, each in enumerate(uniqVals):
            y[y == each] = i
            self.__classes[i] = each

        # Transform the true answers into vectors
        y = np.where(np.arange(len(uniqVals)) == y, 1, 0)

        # Training
        for _ in range(self.numOfLoop):
            h = sigmoid(x_train @ self.__w1)  # (n, k)
            # y_pred = softmax(h @ self.__w2)  # (n, c)
            y_pred = h @ self.__w2  # (n, c)

            # Store loss (for ploting)
            # self.loss.append(crossEntropy(y, y_pred))
            self.loss.append(MSE(y, y_pred))

            w2_gradient = h.T @ (y_pred - y)  # (k, c)
            h_gradient = (y_pred - y) @ self.__w2.T  # (n, k)
            w1_gradient = x_train.T @ (h * (1 - h) * h_gradient)  # (m, k)

            self.__w2 -= w2_gradient * self.__lr  # (k, c)
            self.__w1 -= w1_gradient * self.__lr  # (m, k)

    def predict(self, x):
        h = sigmoid(x @ self.__w1)  # (n, k)
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
    tlnn = TwoLayerNN(10, 0.00001, 8000)
    tlnn.fit(x, y)
    tlnn.predict(x)
