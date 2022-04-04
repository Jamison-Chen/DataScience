import numpy as np
import matplotlib.pyplot as plt


def independent_groups(size=300, group=3, deviation=16, plot=True):
    groupSize = size // group
    m = np.random.RandomState(17).rand(group, 2) * deviation
    x = np.empty((0, 2), dtype=np.float64)
    y = np.empty((0, 1), dtype=np.int8)
    plt.axis("equal")
    for i, each in enumerate(m):
        std = np.random.uniform(deviation / 10, deviation / 7, (2,))
        corrCoeff = np.random.uniform(-0.5, 0.5)
        cov = np.empty((2, 2), dtype=np.float64)
        np.fill_diagonal(cov, corrCoeff * np.prod(std))
        cov = np.fliplr(cov)
        cov[0][0], cov[1][1] = std[0] ** 2, std[1] ** 2
        group = np.random.multivariate_normal(each, cov, groupSize)
        x = np.concatenate((x, group), axis=0)
        y = np.concatenate((y, np.full(shape=(groupSize, 1), fill_value=i + 1)), axis=0)
    if plot:
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Spectral)
    return x, y
