import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


def distribution(mean, covariance):
    gaussian = np.random.multivariate_normal(mean=mean, cov=covariance, size=500)
    return gaussian


def generate(pi):
    mean1 = [0, 0]
    cov1 = [[1, 0], [0, 1]]
    mean2 = [3, 0]
    cov2 = [[1, 0], [0, 1]]
    mean3 = [0, 3]
    cov3 = [[1, 0], [0, 1]]
    gaussian1 = distribution(mean1, cov1)
    gaussian2 = distribution(mean2, cov2)
    gaussian3 = distribution(mean3, cov3)
    return pi[0] * gaussian1 + pi[1] * gaussian2 + pi[2] * gaussian3


def L2_distance_square(x, y):
    return sum(np.power(x-y, 2))


def objective_function(centroids, assignments, data):
    objective = 0
    for i, d in enumerate(data):
        objective += L2_distance_square(d, centroids[assignments[i]])
    return objective


def KMeans(data, k=5, epochs=20):
    centroids = np.random.choice(data.shape[0], size=k)
    centroids = data[centroids]
    assignment = np.zeros((data.shape[0], ), dtype=np.int8)
    objectives = list()
    for epoch in range(epochs):
        for i, d in enumerate(data):
            # TODO: RE-ASSIGNMENT
            min_distance = float("inf")
            for j, c in enumerate(centroids):
                cur_distance = L2_distance_square(c, d)
                if cur_distance < min_distance:
                    min_distance = cur_distance
                    assignment[i] = j

        # TODO: UPDATE CENTROIDS
        for k_ in range(k):
            pointsInCluster = np.nonzero(k_ == assignment)[0]
            if pointsInCluster.size == 0:
                break
            pointsInCluster = data[pointsInCluster]
            centroids[k_] = pointsInCluster.mean(axis=0)
        objectives.append(objective_function(centroids, assignment, data))
    return centroids, assignment, objectives


if __name__ == "__main__":
    data = generate([0.2, 0.5, 0.3])
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    plt.figure()
    for k in range(2, 6):

        centroids, assignment, objectives = KMeans(data, k=k)

        plt.title("objective of kmeans")
        plt.xlabel("number of iters")
        plt.ylabel('objective value')
        plt.plot(objectives, linewidth=2, label="objective" + str(k))
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.grid()
    plt.legend()
    plt.show()

    for k in range(3, 6, 2):
        centroids, assignment, _ = KMeans(data, k=k)
        # TODO: PLOT
        plt.figure()
        for i in range(k):
            data_cluster = data[np.nonzero(assignment == i)[0]]
            plt.scatter(data_cluster[:, 0], data_cluster[:, 1], marker="o")
        plt.show()
