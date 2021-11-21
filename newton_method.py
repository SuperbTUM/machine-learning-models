import numpy as np
from logistic_regression import load_data, sigmoid, target_function, pretreatment
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


def Jacobian(w, y, x):
    return (1 - sigmoid(np.dot(x, w) * y)) * y * x


def Hessian(w, y, x):
    x_col = x.reshape((x.shape[0], 1))
    x_row = x.reshape((1, x.shape[0]))
    X_square = x_col @ x_row
    y_square = y ** 2
    operator = np.exp(-y * np.dot(x, w))
    mid = -operator / ((1 + operator) ** 2)
    return y_square * mid * X_square


def newton(path, n=10, iteration=100):
    dataset = load_data(path)
    dataset = pretreatment(dataset)
    kf = KFold(n_splits=n, random_state=1, shuffle=True)
    ori_features, ori_gt = dataset[:, :-1], dataset[:, -1]
    losses = np.zeros((n, iteration))
    truth_table = np.zeros((2, 2))
    for ii, (train_idx, test_idx) in enumerate(kf.split(dataset)):
        X_train, y_train = ori_features[train_idx], ori_gt[train_idx]
        X_test, y_test = ori_features[test_idx], ori_gt[test_idx]
        dims = X_train.shape[1]
        samples = X_train.shape[0]
        test_samples = X_test.shape[0]
        w = np.zeros((dims,))
        for iter in range(iteration):
            jacob = np.zeros((dims, ))
            hessian = np.zeros((dims, dims))
            for i in range(samples):
                jacob += Jacobian(w, y_train[i], X_train[i])
                hessian += Hessian(w, y_train[i], X_train[i])
            w = w - np.linalg.inv(hessian) @ jacob

            total_loss = 0.
            for j in range(samples):
                total_loss += target_function(X_train[j], w, y_train[j])

            losses[ii, iter] = total_loss
        for idx in range(test_samples):
            if np.dot(X_test[idx], w) > 0:
                # y_pred = 1
                if y_test[idx] == 1:
                    truth_table[1][1] += 1
                else:
                    truth_table[0][1] += 1
            else:
                # y_pred = -1
                if y_test[idx] == 1:
                    truth_table[1][0] += 1
                else:
                    truth_table[0][0] += 1
    print(truth_table, (truth_table[0][0] + truth_table[1][1])/4600)
    plt.xlabel("iteration")
    plt.ylabel("loss")
    for j in range(losses.shape[0]):
        plt.plot(losses[j], linewidth=2, label="loss of " + str(j) + "th training")
    plt.legend()
    plt.grid()
    plt.savefig("newton_loss.jpg")
    plt.show()


if __name__ == "__main__":
    newton("Bayes_classifier", iteration=100)
