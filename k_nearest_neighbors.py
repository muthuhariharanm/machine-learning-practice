import numpy as np 
from math import sqrt
import matplotlib.pyplot as plt
from matplotlib import style
from collections import Counter
import warnings
import pandas as pd 
import random

# plot1 = [1,3]
# plot2 = [2,5]
# euclidean_distance = sqrt( (plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2 )

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is less than data')
    
    distances = []
    for group in data:
        for features in data[group]:
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict)) 
            distances.append([euclidean_distance,group])


    votes = [ i[1] for i in sorted(distances)[:k] ]
    #print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]

    return vote_result

# sample dataset
#dataset = {'k': [[1,2],[2,3],[3,1]] , 'r' : [[6,5],[7,7],[8,6]]}

# making dataset
df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?',-99999, inplace = True)
df.drop(['id'], 1, inplace = True)
data = df.astype(float).values.tolist()
random.shuffle(data)

# splitting it to train and test
test_size = 0.2
train_set = {2 : [], 4 : []}
test_set = {2 : [], 4 : []}
train_data = data[ : -int(test_size*len(data))]
test_data = data[-int(test_size*len(data)) : ]

# populate the dictionaries
for i in train_data:
    train_set[i[-1]].append(i[:-1])
for i in test_data:
    test_set[i[-1]].append(i[:-1])


# Calculating accuracy of the model
correct = 0
total = 0
for group in test_set:
    for dt in test_set[group]:
        vote = k_nearest_neighbors(train_set,dt,5)
        if group == vote:
            correct += 1
        total += 1

print('Accuracy : ', correct / total)


# target data
# #new_feature = [5,7]
new = [4,2,3,1,2,1,1,3,2]

# for i in dataset:
#     for ii in dataset[i]:
#         plt.scatter(ii[0],ii[1],color = i)

# # predicting KNN
result = k_nearest_neighbors(train_set, new, 5)
print('Class : ',result)

# plt.scatter(new_feature[0] , new_feature[1], color = 'g')
# #plt.show()