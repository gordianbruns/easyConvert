import math
from imageProcessing import *
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # disable tensorflow info and warnings; change to 0 if you want to see them
import tensorflow as tf


# helper function: computes euclidean distance
def dist(x: list, y: list) -> float:
    return math.sqrt(sum(
        [(int(x_i) - int(y_i)) ** 2 for x_i, y_i in zip(x, y)]
    ))


# helper function: get distances to classified neighbors
def get_training_distances(test_sample: list, x_train: np.ndarray) -> list:
    return [dist(simplify_list(train_sample), simplify_list(test_sample)) for train_sample in x_train]


# helper function: determine most frequent element
def get_most_frequent(l: list) -> str:
    return max(l, key=l.count)