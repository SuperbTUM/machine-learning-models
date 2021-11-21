import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

TINY = 1e-6


def normal_distribution(x, mean, sigma):
    # return np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi) * sigma)
    return multivariate_normal.pdf(x, mean=mean, cov=sigma, allow_singular=True)


def load_dataset(path, train=True):
    if train:
        X = pd.read_csv(path + "Prob3_Xtrain.csv", header=None)
        y = pd.read_csv(path + "Prob3_ytrain.csv", header=None)
    else:
        X = pd.read_csv(path + "Prob3_Xtest.csv", header=None)
        y = pd.read_csv(path + "Prob3_ytest.csv", header=None)
    X = X.values
    y = y.values
    return X, y


def EStep(pi, x, mean, sigma):
    phai = np.empty((len(pi), x.shape[0]))  # (k, n)
    for i in range(len(pi)):
        phai[i] = pi[i] * normal_distribution(x, mean[i], sigma[i])
    sum_phai = phai.sum(axis=0)
    phai = phai / sum_phai
    return phai, sum_phai


def UpdatePi(phai, num_samples):
    n = phai.sum(1)
    pi = n / num_samples
    return pi, n


def UpdateMiuCov(phai, X, n, miu, cov):
    for kth in range(phai.shape[0]):
        updated_miu = phai[kth].dot(X) / n[kth]
        miu[kth] = updated_miu

        updated_cov = np.multiply(phai[kth].reshape(-1, 1), (X - miu[kth])).T.dot(
            X - miu[kth]) / n[kth]
        cov[kth] = updated_cov
    return miu, cov


def init(X, order=3):
    pi = np.repeat(1 / order, order)
    sigma = np.repeat(np.cov(X.T).reshape(1, X.shape[1], X.shape[1]), order, axis=0)
    miu = np.mean(X, axis=0)
    miu = np.random.multivariate_normal(miu, np.cov(X.T), order)
    return pi, miu, sigma

# for a cluster of either 0's or 1's, run the training process!


def train(X, iterations=30):
    num_samples = X.shape[0]
    all_objectives = list()
    best_pi = best_miu = best_sigma = None
    best_obj = 0.
    for training_time in range(10):
        pi, miu, sigma = init(X, order=4)
        objectives = list()
        for iter in range(iterations):
            phai, sum_phai = EStep(pi, X, miu, sigma)  # the data itself could be a matrix!
            pi, n = UpdatePi(phai, num_samples)
            miu, sigma = UpdateMiuCov(phai, X, n, miu, sigma)
            cur_obj = objective_function(sum_phai)
            if best_obj < cur_obj:
                best_obj = cur_obj
                best_pi = pi
                best_miu = miu
                best_sigma = sigma
            objectives.append(cur_obj)

        all_objectives.append(objectives)
    return all_objectives, best_pi, best_miu, best_sigma


def objective_function(sum_phai):
    return np.sum(np.log(sum_phai))


def plot_utils(objectives0, objectives1):
    x_axis = [i for i in range(5, 30)]
    plt.figure()
    for i in range(len(objectives0)):
        plt.plot(x_axis, objectives0[i][5:], linewidth=2, label=str(i))
    plt.title("objective for class 0")
    plt.grid()
    plt.legend()
    plt.show()

    plt.figure()
    for j in range(len(objectives1)):
        plt.plot(x_axis, objectives1[j][5:], linewidth=2, label=str(j))
    plt.title("objective for class 1")
    plt.grid()
    plt.legend()
    plt.show()


if __name__ == "__main__":
    train_X, train_y = load_dataset("./", train=True)
    # test_X, test_y = load_dataset("./", train=False)
    indexes_0 = np.nonzero(train_y == 0)[0]
    train_X_0 = train_X[indexes_0]
    objectives0, best_pi0, best_miu0, best_sigma0 = train(train_X_0)

    indexes_1 = np.nonzero(train_y == 1)[0]
    train_X_1 = train_X[indexes_1]
    objectives1, best_pi1, best_miu1, best_sigma1 = train(train_X_1)
    # plot_utils(objectives0, objectives1)
    #
    # print(best_pi0, best_miu0, best_sigma0)
    # print(best_pi1, best_miu1, best_sigma1)

    with open("pi0.npy", "wb") as f:
        np.save(f, best_pi0)
    f.close()

    with open("miu0.npy", "wb") as f:
        np.save(f, best_miu0)
    f.close()

    with open("sigma0.npy", "wb") as f:
        np.save(f, best_sigma0)
    f.close()

    with open("pi1.npy", "wb") as f:
        np.save(f, best_pi1)
    f.close()

    with open("miu1.npy", "wb") as f:
        np.save(f, best_miu1)
    f.close()

    with open("sigma1.npy", "wb") as f:
        np.save(f, best_sigma1)
    f.close()
