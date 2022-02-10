import abc
import numpy as np
from scipy.stats import multivariate_normal


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
        self.__x = x
        self.__y = y

    def predict(self, x):
        result = []
        for each in x:
            # Use "Euclidean Distance" (L2 distance)
            distances = np.linalg.norm(each - self.__x, axis=1)
            indices = np.argpartition(distances, self.__k)[: self.__k]
            uniqueVal, counts = np.unique(
                np.squeeze(self.__y[indices, :]), return_counts=True
            )
            result.append(uniqueVal[counts.argmax()])
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
            self.__cov[classIdx] = np.cov(groupSamples.T)
            if self.__naiveBayes:
                # Assume covariance between different features being 0
                self.__cov[classIdx] *= np.eye(2)
            self.__p[classIdx] = counts[i] / n
        if self.__shareCov:
            self.__meanCov = np.zeros((m, m))
            for classIdx in self.__cov:
                self.__meanCov += self.__p[classIdx] * self.__cov[classIdx]

    def predict(self, x):
        n = x.shape[0]
        pMax = np.zeros((n, 1))
        prediction = np.empty((n, 1), dtype=np.int8)
        for classIdx in self.__mu:
            # Assume Normal Distribution
            pXGivenClass = multivariate_normal.pdf(
                x,
                mean=self.__mu[classIdx],
                cov=self.__meanCov if self.__shareCov else self.__cov[classIdx],
            )
            p = (self.__p[classIdx] * pXGivenClass).reshape(n, 1)
            prediction = np.where(p > pMax, classIdx, prediction)
            pMax = np.maximum(p, pMax)
        return prediction


class LogisticRegression(Model):
    def __init__(self):
        pass

    def fit(self, x, y):
        return super().fit(x, y)

    def predict(self, x):
        return super().predict(x)
