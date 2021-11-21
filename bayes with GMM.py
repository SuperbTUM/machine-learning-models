import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
import numba


def load_dataset(path):
    X = pd.read_csv(path + "Prob3_Xtest.csv", header=None)
    y = pd.read_csv(path + "Prob3_ytest.csv", header=None)
    X = X.values
    y = y.values
    return X, y


def load_config():
    with open("pi0.npy", "rb") as f:
        pi0 = np.load(f)
    f.close()

    with open("miu0.npy", "rb") as f:
        miu0 = np.load(f)
    f.close()

    with open("sigma0.npy", "rb") as f:
        sigma0 = np.load(f)
    f.close()

    with open("pi1.npy", "rb") as f:
        pi1 = np.load(f)
    f.close()

    with open("miu1.npy", "rb") as f:
        miu1 = np.load(f)
    f.close()

    with open("sigma1.npy", "rb") as f:
        sigma1 = np.load(f)
    f.close()

    CONFIG = {"best_pi0": pi0,
              "best_miu0": miu0,
              "best_sigma0": sigma0,
              "best_pi1": pi1,
              "best_miu1": miu1,
              "best_sigma1": sigma1}
    return CONFIG


def bayes_classifier_(pi0, miu0, sigma0, pi1, miu1, sigma1, order, X):
    phai1 = np.zeros((order, X.shape[0]))
    phai0 = np.zeros((order, X.shape[0]))
    for k in range(order):
        phai1[k] = multivariate_normal.pdf(X, miu1[k], sigma1[k], True)
        phai0[k] = multivariate_normal.pdf(X, miu0[k], sigma0[k], True)
    y_pred1 = sum(phai1[k] * pi1[k] for k in range(order))
    y_pred0 = sum(phai0[k] * pi0[k] for k in range(order))
    prediction = np.ones((X.shape[0], ))
    for i in range(X.shape[0]):
        prediction[i] = np.argmax([y_pred0[i], y_pred1[i]])
    return prediction


@numba.njit
def confusion_matrix(prediction, gt):
    TN = TP = FN = FP = 0
    for i in range(len(gt)):
        if prediction[i] == 0 and gt[i] == 0:
            TN += 1
        elif prediction[i] == 1 and gt[i] == 1:
            TP += 1
        elif prediction[i] == 0 and gt[i] == 1:
            FN += 1
        else:
            FP += 1
    return TP, TN, FP, FN


if __name__ == "__main__":
    test_X, test_y = load_dataset("./")
    CONFIG = load_config()
    prediction = bayes_classifier_(CONFIG["best_pi0"],
                      CONFIG["best_miu0"],
                      CONFIG["best_sigma0"],
                      CONFIG["best_pi1"],
                      CONFIG["best_miu1"],
                      CONFIG["best_sigma1"],
                      4,
                      test_X)
    TP, TN, FP, FN = confusion_matrix(prediction, test_y)
    print(TP, TN, FP, FN)
    print("acc =", (TP+TN)/(TP+TN+FP+FN))
