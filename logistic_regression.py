import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


TINY = 1e-5


def load_data(path):
    X = pd.read_csv(path + "/X.csv", header=None)
    y = pd.read_csv(path + "/y.csv", header=None)
    X = np.asarray(X)
    y = np.asarray(y)
    return np.concatenate((X, y), axis=-1)


def sigmoid(input):
    return 1 / (1 + np.exp(-input))


def update(w, y, x, eta=0.01/4600):
    return eta * (1 - sigmoid(np.dot(w, x) * y)) * y * x


def pretreatment(dataset):
    ori_y = dataset[:, -1]
    for i in range(ori_y.shape[0]):
        if ori_y[i] == 0:
            ori_y[i] = -1
    new_dataset = np.ones((dataset.shape[0], dataset.shape[1]+1))
    new_dataset[:, :-2] = dataset[:, :-1]
    new_dataset[:, -1] = ori_y
    return new_dataset


def target_function(x, w, y):

    return np.log(sigmoid(np.dot(x, w) * y) + 1e-6)


def logistic_regression(dataset, n=10, iterations=1000):
    kf = KFold(n_splits=n, random_state=1, shuffle=True)
    plt.figure(figsize=(12, 12))
    losses = np.zeros((n, iterations))
    for ii, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        X_train, y_train = dataset[:, :-1][train_idx], dataset[:, -1][train_idx]
        X_test, y_test = dataset[:, :-1][test_idx], dataset[:, -1][test_idx]
        samples = X_train.shape[0]
        dims = X_train.shape[1]
        w = np.zeros((dims,))
        for iter in range(iterations):
            for idx in range(samples):
                w += update(w, y_train[idx], X_train[idx])
            # y_pred = np.zeros((test_samples, ))
            # for idx in range(test_samples):
            #     if np.dot(X_test[idx], w) > 0:
            #         y_pred[idx] = 1
            #     else:
            #         y_pred[idx] = -1
            total_loss = 0.
            for j in range(samples):
                total_loss += target_function(X_train[j], w, y_train[j])

            losses[ii, iter] = total_loss
    plt.xlabel("iteration")
    plt.ylabel("loss")
    for j in range(losses.shape[0]):
        plt.plot(losses[j], linewidth=2, label="loss of " + str(j) + "th training")
    plt.legend()
    plt.grid()
    plt.savefig("logistic_loss.jpg")
    plt.show()


if __name__ == "__main__":
    dataset = load_data("Bayes_classifier")
    dataset = pretreatment(dataset)

    logistic_regression(dataset, iterations=1000)
