import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numba
from itertools import accumulate


def load_dataset(path):
    X = pd.read_csv(path + "Prob1_X.csv", header=None)
    y = pd.read_csv(path + "Prob1_y.csv", header=None)
    X = np.asarray(X).astype(np.float32)
    y = np.asarray(y).astype(np.int8)
    return X, y


def least_square_classifier(X, y):
    weights = np.linalg.inv(X.T @ X) @ X.T @ y
    return weights


def fetch_classifier(X, weights):  # (n, d) @ (d, 1)
    return np.sign(X @ weights)  # (n, 1)


@numba.njit(nogil=True)
def error(y, classifier, w):
    flip_classifier = False
    e = 0
    for i in range(y.shape[0]):
        if y[i] != classifier[i]:
            e += w[i]
    if e >= 0.5:
        flip_classifier = True
    return e, flip_classifier


def sampling(num_samples, num_sampled, merged, w):
    sampled_dataset_indexes = np.random.choice(num_samples, size=num_sampled, p=w)
    sampled_dataset = merged[sampled_dataset_indexes]
    sampled_X, sampled_y = sampled_dataset[:, :-1], sampled_dataset[:, -1]
    return sampled_X, sampled_y


def train(X, y, iterations=2500):
    # Merge the dataset
    merged = np.hstack((X, y))
    num_samples, dims = X.shape
    y = y.reshape((-1, ))
    classifiers = list()
    alphas = list()
    errors = list()
    epsilons = list()
    distributions = list()
    # Sampling
    num_sampled = 200
    w = np.asarray([1 / num_samples for _ in range(num_samples)])
    distributions.append(w)
    sampled_X, sampled_y = sampling(num_samples, num_sampled, merged, w)
    # INIT
    weights = least_square_classifier(sampled_X, sampled_y)
    classifier = fetch_classifier(X, weights)
    e, flip_classifier = error(y, classifier, w)

    if flip_classifier:
        weights = -weights
        classifier = fetch_classifier(X, weights)
        e, _ = error(y, classifier, w)
    epsilons.append(e)
    for t in range(1, iterations+1):
        alpha = 1 / 2 * np.log((1 - e) / e)
        alphas.append(alpha)
        classifiers.append(classifier)
        training_error = boost_train(classifiers, alphas, y, t)
        errors.append(training_error)
        w *= np.exp(-alpha * y * classifier)
        w /= w.sum(keepdims=True)
        distributions.append(w)
        assert np.isclose(sum(w), 1)
        sampled_X, sampled_y = sampling(num_samples, num_sampled, merged, w)

        # INIT
        weights = least_square_classifier(sampled_X, sampled_y)
        classifier = fetch_classifier(X, weights)
        e, flip_classifier = error(y, classifier, w)

        if flip_classifier:
            weights = -weights
            classifier = fetch_classifier(X, weights)
            e, _ = error(y, classifier, w)
        epsilons.append(e)
        if t % 100 == 0:
            print("iteration", t)
    alphas = np.asarray(alphas)
    classifiers = np.asarray(classifiers)
    errors = np.asarray(errors)
    epsilons = np.asarray(epsilons).astype(np.float32)
    distributions = np.asarray(distributions).astype(np.float32)
    return alphas, classifiers, errors, epsilons, distributions


def cal_upper_bound(errors):
    # accum = np.asarray(list(accumulate((1/2 - errors) ** 2)))
    # return np.exp(-2 * accum)
    Zt = 1.
    Zts = []
    for i in range(errors.shape[0]):
        Zt *= np.sqrt(1 - 4 * (1/2 - errors[i]) ** 2)
        Zts.append(Zt)
    return Zts


@numba.njit(nogil=True)
def boost_train_utils(alphas, classifiers):
    final_classifiers = np.empty((y.shape[0],), dtype=np.float32)
    for j in range(y.shape[0]):
        final_classifier = 0.
        for i in range(len(alphas)):
            final_classifier += alphas[i] * classifiers[i, j]
        final_classifier = np.sign(final_classifier)
        final_classifiers[j] = final_classifier
    return final_classifiers


def boost_train(classifiers, alphas, y, t):
    assert len(classifiers) == t
    classifiers = np.asarray(classifiers)
    alphas = np.asarray(alphas)
    final_classifiers = boost_train_utils(alphas, classifiers)
    e, _ = error(y, final_classifiers, w=np.ones((y.shape[0], )))
    e /= y.shape[0]
    return e


if __name__ == "__main__":
    X, y = load_dataset("./")
    iterations = 2500
    alphas, classifiers, errors, epsilons, distributions = train(X, y, iterations)
    upper_bounds = cal_upper_bound(epsilons)
    plt.figure()
    X_axis = np.arange(iterations)
    plt.plot(X_axis, errors, linewidth=2, label="errors")
    plt.plot(X_axis, upper_bounds[:-1], linewidth=2, label="upper bounds")
    plt.xlabel("iters")
    plt.title("Problem 1.1 plot")
    plt.grid()
    plt.legend()
    plt.show()

    # mean_distribution = np.mean(distributions, axis=0)
    # plt.figure()
    # plt.stem(mean_distribution)
    # plt.title("Problem 1.2 plot")
    # plt.grid()
    # plt.show()
    #
    # plt.figure()
    # plt.plot(alphas, linewidth=2, label="alpha")
    # plt.plot(epsilons, linewidth=2, label="epsilon")
    # plt.xlabel("iterations")
    # plt.ylabel("values")
    # plt.title("Problem 1.3 plot")
    # plt.grid()
    # plt.legend()
    # plt.show()
