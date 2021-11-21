import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(pathX, pathy=None):
    dfX = pd.read_csv(pathX, delimiter=",", header=None)
    if pathy:
        dfy = pd.read_csv(pathy, delimiter=",", header=None)
        return np.array(dfX), np.array(dfy)
    else:
        return np.array(dfX)


def wrr(lamda, X, y):
    features = X.shape[1]
    X = np.matrix(X)
    y = np.matrix(y)
    I = np.matrix(np.identity(features))
    return ((lamda * I + X.T @ X).I @ X.T) @ y


def call_wrr(pathX, pathy, lamda):
    X, y = load_data(pathX, pathy)
    wrr_list = []
    for l in lamda:
        wrr_list.append(wrr(l, X, y))
    wrr_list.append(np.matrix([[0] for _ in range(X.shape[1])]))
    wrr_array = np.array(wrr_list).squeeze()
    return wrr_array


def call_df(X, lamda):
    U, S, V = np.linalg.svd(X)
    df = []
    for l in lamda:
        cur = 0
        for i in range(len(S)):
            cur += (S[i] ** 2) / (l + S[i] ** 2)
        df.append(cur)
    df.append(0)
    return df


def predict(X, wrr_list, lamda):
    res = []
    for i, l in enumerate(lamda):
        res.append(X @ wrr_list[i])
    return np.array(res)


def call_rmse(predy, truthy):
    length = predy.shape[0]
    rmse = np.sqrt(np.mean((predy.reshape((length, 1))-truthy)**2))
    return rmse


def standardization(X):
    mean = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    return (X - mean) / sigma


def lp_regression(X, y, lamda, p):
    w_ls_list = []
    new_X = standardization(X[:, :-1])
    for i in range(2, p+1):
        X_deal = X[:, :-1] ** i
        for j in range(X_deal.shape[1]):
            X_deal = standardization(X_deal)
        new_X = np.concatenate((new_X, X_deal), axis=1)

    new_X = np.concatenate((np.ones([X.shape[0], 1]), new_X), axis=1)
    for l in lamda:
        w_rr = wrr(l, new_X, y)
        w_ls_list.append(w_rr)

    return w_ls_list


def poly_predict(X_test, y, wrr_list, lamda, p):
    new_X_test = standardization(X_test[:, :-1])

    for i in range(2, p+1):
        X_test_deal = X_test[:, :-1] ** i
        for j in range(X_test_deal.shape[1]):
            X_test_deal = standardization(X_test_deal)
        new_X_test = np.concatenate((new_X_test, X_test_deal), axis=1)
    new_X_test = np.concatenate((np.ones([X_test.shape[0], 1]), new_X_test), axis=1)
    y_predict = predict(new_X_test, wrr_list, lamda).squeeze()

    rmse_list = []
    for l in lamda:
        rmse_list.append(call_rmse(y_predict[l], y))

    return rmse_list


if __name__ == '__main__':
    pathX = 'hw1-data/X_train.csv'
    pathy = 'hw1-data/y_train.csv'
    X, y = load_data(pathX, pathy)
    lamda = [i for i in range(5001)]
    wrr_list = call_wrr(pathX, pathy, lamda)
    df = call_df(X, lamda)
    labels = ["cylinder", "displacement", "horsepower", "weight", "acceleration", "year made", "reference"]
    plt.figure()
    for j in range(wrr_list.shape[1]):
        plt.plot(df, wrr_list[:, j], linewidth=2, label=labels[j])
    plt.legend()
    plt.xlabel("df(lambda)")
    plt.ylabel("w with respect to their features")
    plt.grid()
    plt.show()
    testX = "hw1-data/X_test.csv"
    testy = "hw1-data/y_test.csv"
    lamda_test = [k for k in range(51)]
    X_test, y_test = load_data(testX, testy)
    wrr_pred = predict(X_test, wrr_list, lamda_test)
    rmse_list = []
    for i in range(51):
        rmse = call_rmse(wrr_pred[i], y_test)
        rmse_list.append(rmse)
    plt.figure()
    plt.plot(lamda_test, rmse_list, linewidth=2)
    plt.xlabel("lambda")
    plt.ylabel("rmse")
    plt.grid()
    plt.show()
    lamda_poly = [i for i in range(101)]
    w_rr_2 = lp_regression(X, y, lamda_poly, p=2)
    w_rr_3 = lp_regression(X, y, lamda_poly, p=3)
    rmse2 = poly_predict(X_test, y_test, w_rr_2, lamda_poly, p=2)
    rmse3 = poly_predict(X_test, y_test, w_rr_3, lamda_poly, p=3)
    rmse_list_new = []
    lamda_test_new = [k for k in range(101)]
    wrr_pred = predict(X_test, wrr_list, lamda_test_new)
    for i in range(101):
        rmse = call_rmse(wrr_pred[i], y_test)
        rmse_list_new.append(rmse)
    plt.figure()
    plt.plot(lamda_poly, rmse_list_new, linewidth=2, label="p=1")
    plt.plot(lamda_poly, rmse2, linewidth=2, label="p=2")
    plt.plot(lamda_poly, rmse3, linewidth=3, label="p=3")
    plt.xlabel("lambda")
    plt.ylabel("RMSE")
    plt.legend()
    plt.grid()
    plt.show()
    print(np.argmin(np.array(rmse3)))