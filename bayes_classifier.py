import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def load_data(path):
    X = pd.read_csv(path + "/X.csv", header=None)
    y = pd.read_csv(path + "/y.csv", header=None)
    X, y = np.asarray(X), np.asarray(y)
    return np.concatenate((X, y), axis=-1)


def shuffle(dataset):
    idxes = np.random.permutation(dataset.shape[0])
    shuffled = dataset[idxes]
    return shuffled


def prob(pi, lamda, x, k=0):
    if k == 1:
        base = pi
        for i in range(x.shape[0]):
            base *= (np.exp(-lamda[i]) * lamda[i] ** x[i])
        return base
    else:
        base = 1 - pi
        for i in range(x.shape[0]):
            base *= (np.exp(-lamda[i]) * lamda[i] ** x[i])
        return base


def calculate_pi(y):
    return y.mean()


def calculate_lamda(X, y):
    yisy0 = np.nonzero(y == 0)[0]
    yisy1 = np.nonzero(y == 1)[0]
    count_y0 = yisy0.shape[0]
    count_y1 = yisy1.shape[0]
    lamda = np.zeros((2, X.shape[1]))
    for i in range(X.shape[1]):
        lamda[0][i] = (X[:, i][yisy0].sum() + 1) / (count_y0 + 1)
        lamda[1][i] = (X[:, i][yisy1].sum() + 1) / (count_y1 + 1)
    return lamda


def plot_lambdas(lamdas):
    plt.figure(figsize=(12, 12))
    x = [i for i in range(lamdas.shape[1])]
    plt.title("stem plot of lambda")
    plt.xlabel("dimension of x")
    plt.ylabel("lambda value")
    plt.stem(x, lamdas[0, :], markerfmt="o", basefmt="-", label="k=0")
    plt.legend()
    plt.grid()
    plt.savefig("stem0.jpg")
    plt.show()

    plt.figure(figsize=(12, 12))
    plt.title("stem plot of lambda")
    plt.xlabel("dimension of x")
    plt.ylabel("lambda value")
    plt.stem(x, lamdas[1, :], markerfmt="o", basefmt="-", label="k=1")
    plt.legend()
    plt.grid()
    plt.savefig("stem1.jpg")
    plt.show()


def naive_bayes(dataset, n=10):
    forum_all = np.zeros((2, 2))
    kf = KFold(n_splits=n)
    sum_lambda = np.zeros((2, dataset.shape[1]-1))
    for train_idx, test_idx in kf.split(dataset):
        y_train = dataset[:, -1][train_idx]
        X_train = dataset[:, :-1][train_idx]
        y_test = dataset[:, -1][test_idx]
        X_test = dataset[:, :-1][test_idx]
        pi = calculate_pi(y_train)
        lamdas = calculate_lamda(X_train, y_train)
        sum_lambda += lamdas
        y_pred = np.zeros((X_test.shape[0], ))
        forum = np.zeros((2, 2))
        for i in range(X_test.shape[0]):
            prob_pos = prob(pi, lamdas[1, :], X_test[i, :], k=1)
            prob_neg = prob(pi, lamdas[0, :], X_test[i, :], k=0)
            y_pred[i] = 1 if prob_pos > prob_neg else 0
            if y_pred[i] == 1 and y_test[i] == 1:
                forum[1][1] += 1
            elif y_pred[i] == 0 and y_test[i] == 0:
                forum[0][0] += 1
            elif y_pred[i] == 1 and y_test[i] == 0:
                forum[0][1] += 1
            else:
                forum[1][0] += 1
        forum_all += forum
    ave_lambda = sum_lambda / n
    plot_lambdas(ave_lambda)
    return forum_all


if __name__ == "__main__":
    dataset = load_data("Bayes_classifier")
    dataset = shuffle(dataset)
    truth_table = naive_bayes(dataset)
    print(truth_table)
    acc = (truth_table[0][0] + truth_table[1][1]) / 4600
    print(acc)
