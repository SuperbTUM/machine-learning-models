import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tabulate import tabulate


def load_dataset(path):
    nyt_data = pd.read_csv(path, header=None, delimiter=" ")
    data = np.zeros((3012, 8447))
    for index, row in nyt_data.iterrows():
        content = row[0].split(",")
        for words in content:
            word_index, freq = words.split(":")
            word_index = int(word_index) - 1
            data[word_index, index] += int(freq)
    return data


def init():
    W = np.random.uniform(1, 2, (3012, 25))
    H = np.random.uniform(1, 2, (25, 8447))
    return W, H


def normalize_W(W):
    return W / W.sum(axis=0, keepdims=True)


def train(W, H, X, iters=100):
    eps = 1e-16
    losses = list()

    for i in range(iters):
        temp = W @ H
        temp_X = X / (temp + eps)
        Wt = W.T
        temp_W = Wt / (Wt.sum(axis=1, keepdims=True) + eps)
        H *= temp_W @ temp_X

        temp = W @ H
        temp_X = X / (temp + eps)
        Ht = H.T
        temp_H = Ht / (Ht.sum(axis=0, keepdims=True) + eps)
        W *= temp_X @ temp_H

        losses.append(np.sum(-X * np.log(W @ H+eps) + W @ H))
    return losses


def plot(losses):
    plt.figure()
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.title("objective curve")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    data = load_dataset("nyt_data.txt")
    words = pd.read_csv("nyt_vocab.dat", header=None).values.flatten()
    W, H = init()
    losses = train(W, H, data)
    # plot(losses)
    W_norm = normalize_W(W)
    results = list()
    top_words = np.argsort(W_norm, axis=0)
    top_weights = np.sort(W_norm, axis=0)
    df = pd.DataFrame()
    for i in range(25):
        top_10 = words[top_words[:, i][::-1][:10]]
        top_10_weights = top_weights[:, i][::-1][:10].astype(np.float16)
        combo = list(zip(top_10, top_10_weights))
        df["Topic" + str(i)] = combo
    print(tabulate(df, headers='keys', tablefmt='github'))
