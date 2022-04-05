from imageProcessing import process_image, read_images, read_labels, extract_features
from ocrAlgorithms import knn
from helpers.common import *

DEBUG = True

if DEBUG:
    from PIL import Image
    import numpy as np


    def read_image(path):
        return np.asarray(Image.open(path).convert('L'))


    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)

DATA_DIR = 'data/'
TEST_DIR = 'test/'
TEST_DATA_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_DATA_LABEL = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_DATA_LABEL = DATA_DIR + 'train-labels.idx1-ubyte'


def run(argv):
    process_image(argv[0])
    x_train = read_images(TRAIN_DATA_FILENAME, 10000)
    y_train = read_labels(TRAIN_DATA_LABEL, 10000)
    x_test = read_images(TEST_DATA_FILENAME, 5)
    y_test = read_labels(TEST_DATA_LABEL, 5)

    if DEBUG:
        # Write some images out just so we can see them visually.
        for idx, test_sample in enumerate(x_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')
        x_test = [read_image(f'{DATA_DIR}our_test.png')]
        y_test = [0]

    x_train = extract_features(x_train)
    x_test = extract_features(x_test)

    y_prediction = knn(x_train, y_train, x_test, y_test, 5)
    print(f'Prediction: {y_prediction}')

    print(f'Expected: {y_test}')
    accuracy = sum([int(y_prediction_i == y_test_i)
                    for y_prediction_i, y_test_i in zip(y_prediction, y_test)]) / len(y_test)

    print(f'Accuracy: {accuracy * 100}%')
