import numpy as np
import matplotlib.pyplot as plt


def testModel(model, x_train, y_train, x_test, y_test):
    train_prediction = model.predict(x_train).reshape(y_train.shape)
    train_accuracy = (train_prediction == y_train).sum() / np.squeeze(y_train).size
    print("Training Accuracy: {:.4f}".format(train_accuracy))
    test_prediction = model.predict(x_test).reshape(y_test.shape)
    test_accuracy = (test_prediction == y_test).sum() / np.squeeze(y_test).size
    print("Testing Accuracy: {:.4f}".format(test_accuracy))


def plotDecisionBoundary(model, x_test, y_test):
    # Set min and max values and give it some padding
    x_min, x_max = x_test[:, 0].min() - 1, x_test[:, 0].max() + 1
    y_min, y_max = x_test[:, 1].min() - 1, x_test[:, 1].max() + 1
    h = 0.1
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    zz = model.predict(np.c_[xx.ravel(), yy.ravel()])
    # Plot the contour and testing datas
    plt.gca().set_aspect("equal", adjustable="box")
    plt.contourf(xx, yy, zz.reshape(xx.shape), cmap=plt.cm.Spectral)
    plt.scatter(x_test[:, 0], x_test[:, 1], c=y_test, cmap=plt.cm.Spectral)


def plotLoss(model):
    plt.plot(range(1, model.numOfLoop + 1), model.lossHistory)
