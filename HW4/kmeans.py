import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from random import seed
from random import randint
import copy
# read data
data = pd.read_csv("DataForKmeans.csv",header=None).to_numpy()

# calculate statistics
def evaluate(Means, data):
    errs = []
    for i in range(200):
        Mean = np.resize(Means[i,:], [5,2])
        err = 0
        for j in range(500):
            err += np.min(np.linalg.norm(Mean - data[j,:], axis = 1))**2
        errs.append(err)
    err_mean = np.mean(errs)
    err_min = np.min(errs)
    err_std = np.std(errs)
    return err_mean, err_min, err_std

# K-mean
def kmeans(data, k, thre, Mean):
    # Initialize centroids randomly
    centroids = data[np.random.choice(data.shape[0], k, replace=False)]
    prev_centroids = np.zeros(centroids.shape)

    # Loop until convergence or maximum iterations reached
    while np.linalg.norm(centroids - prev_centroids) > thre:
        prev_centroids = centroids.copy()

        # Assign each data point to the nearest centroid
        clusters = [[] for _ in range(k)]
        for point in data:
            distances = [np.linalg.norm(point - centroid) for centroid in centroids]
            cluster_index = np.argmin(distances)
            clusters[cluster_index].append(point)

        # Update centroids based on the mean of points in each cluster
        for i in range(k):
            centroids[i] = np.mean(clusters[i], axis=0)

    # Compute the Mean and return it
    Mean = centroids
    return Mean

k = 5
thre = 0.00001
iter_num = 200
Means = np.zeros([iter_num, 10]) # kmeans

for iter_id in range(iter_num):
    Mean = None
    centroids = kmeans(data, k, thre, Mean)
    Means[iter_id] = centroids.flatten()

err_mean, err_min, err_std = evaluate(Means, data)


def plot_clusters(data, centroids):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c='blue', alpha=0.1)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=100)
    plt.title("K-means Clustering")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()

plot_clusters(data, Means.reshape(-1, 2))