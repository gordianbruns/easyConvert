import math
from helpers.common import *


# helper function to compute euclidean distance
def dist(x, y):
    return math.sqrt(sum(
        [(bytes_to_int(x_i) - bytes_to_int(y_i))**2 for x_i, y_i in zip(x, y)]
    ))


# helper function to get distances to classified neighbors
def get_training_distances(test_sample, x_train):
    return [dist(train_sample, test_sample) for train_sample in x_train]


# helper function to determine most frequent element
def get_most_frequent(l):
    return max(l, key=l.count)


def knn(x_train, y_train, x_test, y_test, k=3):
    y_prediction = []
    count = 0
    for idx, test_sample in enumerate(x_test):
        training_distances = get_training_distances(test_sample, x_train)
        sorted_distance_indices = [
            pair[0] for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1])
            ]
        neighbors = [y_train[index] for index in sorted_distance_indices[:k]]
        if count < 10:
            #print(f'Point is {bytes_to_int(y_test[idx])} and we guessed {neighbors}')
            count += 1
        nearest_neighbor = get_most_frequent(neighbors)
        y_prediction.append(nearest_neighbor)
    return y_prediction
