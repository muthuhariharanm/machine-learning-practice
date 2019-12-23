import matplotlib.pyplot as plt  
import numpy as np 
import pandas as pd
from sklearn.datasets import make_blobs

class MeanShift:
    def __init__(self, bandwidth = None, bandwidth_norm_step=100):
        self.bandwidth = bandwidth
        self.bandwidth_norm_step = bandwidth_norm_step
    
    def fit(self, data):

        if self.bandwidth == None:
            all_data_centroid = np.average(data, axis=0)
            all_data_norm = np.linalg.norm(all_data_centroid)
            self.bandwidth = all_data_norm / self.bandwidth_norm_step

        centroids = {}

        for i in range(len(data)):
            centroids[i] = data[i]

        weights = [i for i in range(self.bandwidth_norm_step)][::-1] 

        while True:
            new_centroids = []
            for i in centroids:
                in_bandwidth = []
                centroid = centroids[i]

                for featureset in data:
                    #if np.linalg.norm(featureset - centroid) < self.bandwidth:
                    #    in_bandwidth.append(featureset)
                    distance = np.linalg.norm(featureset - centroid)
                    if distance == 0:
                        distance = 0.00000000001
                    weight_index = int(distance / self.bandwidth)

                    if weight_index > self.bandwidth_norm_step - 1:
                        weight_index = self.bandwidth_norm_step - 1

                    to_add = (weights[weight_index]**2)*[featureset]
                    in_bandwidth += to_add

                new_centroid = np.average(in_bandwidth, axis=0)
                new_centroids.append(tuple(new_centroid))

            uniques = sorted(list(set(new_centroids)))

            to_pop = []

            for i in uniques:
                for ii in uniques:
                    if i == ii:
                        pass
                    elif np.linalg.norm(np.array(i) - np.array(ii)) <= self.bandwidth:
                        to_pop.append(ii)
                        break
            
            for i in to_pop:
                try:
                    uniques.remove(i)
                except:
                    pass
            
            prev_centroids = dict(centroids)

            centroids = {}

            for i in range(len(uniques)):
                centroids[i] = np.array(uniques[i])

            optimized = True

            for i in centroids:
                if not np.array_equal(centroids[i], prev_centroids[i]):
                    optimized = False
                if not optimized:
                    break

            if optimized:
                break
            
        self.centroids = centroids

        self.classifications = {}

        for i in range(len(self.centroids)):
            self.classifications[i] = []

        for featureset in data:
            distances = [ np.linalg.norm(featureset - self.centroids[centroid]) for centroid in self.centroids ]
            classification = distances.index(min(distances))
            self.classifications[classification].append(featureset)

    def predict(self, data):
        distances = [ np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids ]
        classification = distances.index(min(distances))
        return classification

# Sample dataset
X = np.array([[1, 2],
              [1.5, 1.8],
              [5, 8 ],
              [8, 8],
              [1, 0.6],
              [9,11],
              [8,2],
              [10,2],
              [9,3],])


#plt.scatter(X[:,0], X[:,1])


colors = ['r','g','b','y','o']

# Classifying and finding Centroids
clf = MeanShift()

clf.fit(X)
centroids = clf.centroids

for classification in clf.classifications:
    for featureset in clf.classifications[classification]:
        plt.scatter(featureset[0], featureset[1], color = colors[classification])

for centroid in centroids:
    plt.scatter(centroids[centroid][0], centroids[centroid][1], marker='x')

# Target dataset
Y = np.array([
    [2,3],
    [4,6],
    [5,7],
    [3,6],
    [5,5],
    [9,1],
    [8,2]
])

for y in Y:
    test_result = clf.predict(y)
    plt.scatter(y[0], y[1], color = colors[test_result])

plt.show()