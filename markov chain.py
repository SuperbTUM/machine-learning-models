import pandas as pd
import numpy as np
from scipy.sparse.linalg import eigs
from scipy.linalg import eig
from tabulate import tabulate
import matplotlib.pyplot as plt


def load_dataset(path):
    data = pd.read_csv(path, header=None)
    return data.values


def transition_matrix(data):
    M = np.zeros((769, 769))
    is_A_win = lambda x, y: 1 if x > y else 0
    is_B_win = lambda x, y: 1 if x < y else 0
    for (teamA_index, teamA_score, teamB_index, teamB_score) in data:
        teamA_index -= 1
        teamB_index -= 1
        M[teamA_index, teamA_index] += is_A_win(teamA_score, teamB_score) + teamA_score / (teamA_score + teamB_score)
        M[teamB_index, teamB_index] += is_B_win(teamA_score, teamB_score) + teamB_score / (teamA_score + teamB_score)
        M[teamA_index, teamB_index] += is_B_win(teamA_score, teamB_score) + teamB_score / (teamA_score + teamB_score)
        M[teamB_index, teamA_index] += is_A_win(teamA_score, teamB_score) + teamA_score / (teamA_score + teamB_score)
    M = M / M.sum(axis=1, keepdims=True)
    return M


def ranking(w, M):
    steps = [10, 100, 1000, 10000]
    records = list()
    w_list = list()
    for i in range(max(steps)+1):
        w = w @ M
        # w = M @ w
        if i in steps:
            records.append(w)
        w_list.append(w)
    return records, w_list


def cal_diff(M):
    eps = 1e-16
    for _ in range(15):
        e_value, e_vector = eigs(M.T, 1)
    u = e_vector.T
    w_inf = u / (u.sum() + eps)
    diff = list()
    w0 = np.repeat(1/769, 769)
    for i in range(10000):
        w0 = np.dot(w0, M)
        diff.append(np.sum(np.abs(w_inf - w0)))
    return diff


def plot_diff(diff):
    figure = plt.figure()
    plt.title("difference between w_inf and w_t")
    plt.plot(diff)
    plt.xlabel("time step")
    plt.ylabel("diff")
    plt.grid()
    plt.show()
    figure.savefig("hw4_q1_new.png")


if __name__ == "__main__":
    data = load_dataset("CFB2019_scores.csv")
    M = transition_matrix(data)
    w = np.full((1, 769), 1/769)
    records, w_list = ranking(w, M)
    names = pd.read_csv("TeamNames.txt", header=None)
    names = names.values.flatten()
    df = pd.DataFrame()
    for i, record in enumerate(records):
        values = np.sort(record.flatten())[::-1][:25]
        record = np.argsort(record.flatten())[::-1]
        first_25 = record[:25]
        key_value = list(zip(names[first_25], values.astype(np.float16)))
        df = df.append({"Time step": 10**(i+1), "ranking": key_value}, ignore_index=True)
        # print("time step: {:.0f}, ranking: {}".format(10**(i+1), key_value))
    print(tabulate(df, headers='keys', tablefmt='github'))
    wd = cal_diff(M)
    plot_diff(wd)
