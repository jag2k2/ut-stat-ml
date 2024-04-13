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

# Function for inverse transform sampling
def inverse_transform_sampling(p, u):
    cumulative_prob = np.cumsum(p)
    # new_index = np.argmax(cumulative_prob > u)
    # new_index = np.random.choice(len(data), p = p)
    Mean = np.searchsorted(cumulative_prob, u)
    return Mean

def kmeans_plusplus_initial(data, k):
    centroids = []

    first_center_idx = np.random.choice(data.shape[0])                  # Randomly choose the first centroid from the data points
    centroids.append(data[first_center_idx])

    for _ in range(1, k):
        d_squared = []
        for point in data:                                              # for each point in data, find its squared distance from the closest known centroid
            min_distance = np.inf
            for centroid in centroids:
                distance = np.linalg.norm(point - centroid)
                if distance < min_distance:
                    min_distance = distance
            d_squared.append(min_distance**2)

        probabilities = np.array(d_squared) / np.sum(d_squared)         # normallize the squared distances to obtain probabilities
        u = np.random.rand()
        selected_index = inverse_transform_sampling(probabilities, u)   # select new centroid using square distance as a probability
        selected_point = data[selected_index]
        centroids.append(selected_point)
    return np.array(centroids)

k = 5
thre = 0.00001
iter_num = 200
Means = np.zeros([iter_num, 10])                                          # Initialize 1D numpy array that holds the centroids

for iter_id in range(iter_num):
    number_of_rows = data.shape[0]                                        # Data has 500 two diminesional data points
    Mean = kmeans_plusplus_initial(data, k)
    centroid_list = kmeans(data, k, thre, Mean)                           # Calculate the kmeans of the data set
    Means[iter_id] = centroid_list.flatten()                                  

err_mean, err_min, err_std = evaluate(Means, data)                        


def plot_clusters(data, means):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c='red', marker = 'x', alpha=0.25, label='data')
    plt.scatter(means[:, 0], means[:, 1], c='blue', marker = 'o', s=10, label='centroids')
    plt.title("K-means++ Clustering")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.show()

centroids = Means.reshape(-1, 2)
plot_clusters(data, centroids)

print("err_mean: ", err_mean)
print("err_min: ", err_min)
print("err_std: ", err_std)