import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(path, train=True):
    if train:
        X = pd.read_csv(path + "/X_train.csv", header=None)
        y = pd.read_csv(path + "/y_train.csv", header=None)
    else:
        X = pd.read_csv(path + "/X_test.csv", header=None)
        y = pd.read_csv(path + "/y_test.csv", header=None)
    X = np.asarray(X)
    y = np.asarray(y)
    return X, y


def kernel(x, y, b):
    return np.exp(-((x-y) ** 2).sum() / b)


def calculate_mean(X, y, x0, b, mid):
    N, d = X.shape
    left = np.zeros((N, 1))
    for i in range(N):
        left[i] = kernel(X[i], x0, b)
    left = left.T
    for i in range(N):
        for j in range(N):
            mid[i][j] += kernel(X[i], X[j], b)

    mid = np.linalg.inv(mid)
    y = y.reshape((N, ))
    return left.dot(mid).dot(y)


def RMSE(preds, y_test):
    return np.sqrt(((preds - y_test) ** 2).mean())


def task_a():
    X_train, y_train = load_data("Gaussian_process")
    X_test, y_test = load_data("Gaussian_process", False)
    bs = [5, 7, 9, 11, 13, 15]
    sigmas = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.]
    the_RMSE = []
    least = 100.
    best_b = 0.
    best_sigma = 0.

    for b in bs:
        for sigma in sigmas:
            y_preds = []
            mid = np.ones((X_train.shape[0],))
            mid = np.diag(mid)
            mid *= sigma
            for i in range(X_test.shape[0]):
                y_preds.append(calculate_mean(X_train, y_train, X_test[i], b, mid.copy())[0])
            y_preds = np.asarray(y_preds).reshape((-1, 1))
            the_RMSE.append([b, sigma, RMSE(y_preds, y_test)])
            if least > the_RMSE[-1][-1]:
                least = the_RMSE[-1][-1]
                best_b = the_RMSE[-1][0]
                best_sigma = the_RMSE[-1][1]
    print(best_b, best_sigma, least)
    print(the_RMSE)


def taskb():
    plt.figure(figsize=(12, 12))
    X_train, y_train = load_data("Gaussian_process")
    X_test, y_test = load_data("Gaussian_process", False)
    X_train, X_test = X_train[:, 3], X_test[:, 3]
    plt.scatter(X_train, y_train, label="dataset")
    X_train = X_train.reshape(-1, 1)
    X_test = X_test.reshape(-1, 1)
    y_preds = []
    mid = np.ones((X_train.shape[0],))
    mid = np.diag(mid)
    mid *= 2
    for i in range(X_train.shape[0]):
        y_preds.append(calculate_mean(X_train, y_train, X_train[i], 5, mid.copy())[0])
    y_preds = np.asarray(y_preds)
    new_list = list(zip(X_train.flatten(), y_preds))
    new_list.sort()
    plt.plot([the_list[0] for the_list in new_list], [the_list[1] for the_list in new_list], color="r", label="predictive_mean")
    plt.legend()
    plt.grid()
    plt.savefig("final.jpg")
    plt.show()


if __name__ == "__main__":
    task_a()
