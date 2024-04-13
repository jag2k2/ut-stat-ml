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
    centroids = Mean
    prev_centroids = np.zeros(centroids.shape)

    while np.linalg.norm(centroids - prev_centroids) > thre:            # Check to see if centroids have converged
        prev_centroids = centroids.copy()

        clusters = [[] for _ in range(k)]                               # creates k empty lists for classification
        for point in data:                                              # for each data point
            distances = []
            for centroid in centroids:                                  # and for each centroid
                distance = np.linalg.norm(point - centroid)             # calculate the point's distance from each centroid
                distances.append(distance)
            centroid_index = np.argmin(distances)                        # find the closest centroid
            clusters[centroid_index].append(point)                       # assign the point to that cluster

        for i in range(k):
            centroids[i] = np.mean(clusters[i], axis=0)                 # Update each centroid with the mean of each cluster

    return centroids

k = 5
thre = 0.00001
iter_num = 200
Means = np.zeros([iter_num, 10])                                          # Initialize 1D numpy array that holds the centroids

number_of_rows = data.shape[0]                                        # Data has 500 two diminesional data points
Mean = data[np.random.choice(number_of_rows, k, replace=False)]       # Randomly select k data points as initial cluster means

for iter_id in range(iter_num):

    centroid_list = kmeans(data, k, thre, Mean)                           # Calculate the kmeans of the data set
    Means[iter_id] = centroid_list.flatten()                                  

err_mean, err_min, err_std = evaluate(Means, data)                        


def plot_clusters(data, means):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c='red', marker = 'x', alpha=0.25, label='data')
    plt.scatter(means[:, 0], means[:, 1], c='blue', marker = 'o', s=10, label='centroids')
    plt.title("K-means Clustering")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

centroids = Means.reshape(-1, 2)
plot_clusters(data, centroids)