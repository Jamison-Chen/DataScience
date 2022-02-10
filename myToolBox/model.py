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


class NaiveBayesClassifier(Model):
    def __init__(self):
        self.__mu = {}
        self.__cov = {}
        self.__p = {}

    def fit(self, x, y):
        size = x.shape[0]
        uniqueVals = np.unique(y)
        for each in uniqueVals:
            groupSamples = x[np.where(y.squeeze() == each)[0], :]
            self.__mu[each] = np.mean(groupSamples, axis=0)
            self.__cov[each] = np.cov(groupSamples.T)
            self.__p[each] = groupSamples.shape[0] / size

    def predict(self, x):
        pMax = np.zeros((x.shape[0], 1))
        prediction = np.empty((x.shape[0], 1), dtype=np.int8)
        for each in self.__mu:
            pXGivenClass = multivariate_normal.pdf(
                x, mean=self.__mu[each], cov=self.__cov[each]
            )
            p = (self.__p[each] * pXGivenClass).reshape(pXGivenClass.size, 1)
            prediction = np.where(p > pMax, each, prediction)
            pMax = np.maximum(p, pMax)
        return prediction
