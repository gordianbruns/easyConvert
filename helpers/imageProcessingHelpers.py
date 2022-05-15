import cv2
import matplotlib.pyplot as plt
import numpy as np
from helpers.common import *
from PIL import Image
import warnings
warnings.filterwarnings("default", category=DeprecationWarning, module=__name__)    # enable DeprecationWarning


# helper function: changes different kinds of lists into one dimensional list
def simplify_list(l: list):
    try:
        # if list is already one dimensional
        if type(l[0]) is np.uint8:
            return l
        # if list is one dimensional and saves all rgb values as sublists
        elif type(l[0]) is list or type(l[0]) is np.ndarray and len(l[0]) == 3:
            # uses only the red value of each pixel; note that the red, blue, and green value are all the same due to
            # binarization
            return [element[0] for element in l]
        else:
            return [pixel for element in l for pixel in element]
    # only for debugging in case new data has a different shape
    except:
        print(type(l[0]))
        raise NotImplementedError(f'type of each element is {type(l[0])}')


# helper function: extract features
def extract_features(x_data: np.ndarray) -> list:
    return [simplify_list(sample) for sample in x_data]