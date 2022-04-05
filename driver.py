from imageProcessing import *
from ocrAlgorithms import knn, neural_network_classifier
from emnist import extract_training_samples, extract_test_samples
from helpers.common import *
import os
from PIL import Image
import time
import pytesseract

DEBUG = True

if DEBUG:
    import numpy as np


    def read_image(path):
        image = Image.open(path).convert('L')
        image = image.resize((32, 32))
        image = add_borders(image)
        return np.asarray(image)


    def write_image(image, path):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)


ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

DATA_DIR = 'data/'
CHAR_DIR = 'char/'
TEST_DIR = 'test/'
TEST_DATA_DIGITS_FILENAME = DATA_DIR + 't10k-images.idx3-ubyte'
TEST_DATA_DIGITS_LABEL = DATA_DIR + 't10k-labels.idx1-ubyte'
TRAIN_DATA_DIGITS_FILENAME = DATA_DIR + 'train-images.idx3-ubyte'
TRAIN_DATA_DIGITS_LABEL = DATA_DIR + 'train-labels.idx1-ubyte'
TEST_DATA_CHARS_FILENAME = DATA_DIR + CHAR_DIR + 'emnist-digits-test-images-idx3-ubyte'
TEST_DATA_CHARS_LABEL = DATA_DIR + CHAR_DIR + 'emnist-digits-test-labels-idx1-ubyte'
TRAIN_DATA_CHARS_FILENAME = DATA_DIR + CHAR_DIR + 'emnist-digits-train-images-idx3-ubyte'
TRAIN_DATA_CHARS_LABEL = DATA_DIR + CHAR_DIR + 'emnist-digits-train-labels-idx1-ubyte'


def run(argv):
    process_image(argv[0])

    method = 0  # 0 = neural network

    print("Training with EMNIST dataset . . .")
    x_train, y_train = extract_training_samples('letters')
    x_train = x_train[:5000] if method == 0 else x_train[:5000].astype(str)
    y_train = y_train[:5000] if method == 0 else y_train[:5000].astype(str)
    #y_train = y_train.astype(str)
    #x_train = []
    #x_buffer = list(x_buffer[:50000:2])
    #y_train = list(y_train[:50000:2])
    x_test, y_test = extract_test_samples('letters')
    x_test = x_test[:40000:400]
    y_test = y_test[:40000:400]

    '''for image in x_buffer:
        new_image = Image.fromarray(image)
        new_image = add_borders(new_image)
        new_image = np.asarray(new_image)
        binarized_image = binarize_image(new_image, 1)
        inverted_image = invert_image(binarized_image)
        x_train.append(flatten_list(inverted_image))'''

    print("Finished training with EMNIST dataset . . .")

    '''correct_predictions = 0
    config = r"--psm 10 --oem 3"

    start = time.time()

    for image_index in range(len(x_test)):
        new_image = Image.fromarray(x_test[image_index])
        prediction = pytesseract.image_to_string(new_image, config=config)
        print(prediction)
        if prediction == y_test[image_index]:
            correct_predictions += 1

    end = time.time()

    print(f'Accuracy: {(correct_predictions/len(x_test)) * 100}%')'''

    '''print("Starting training with ETL dataset . . .")
    for letter in ALPHABET:
        print(f"Training {letter} . . .")
        for image_name in os.listdir(f'extract/{letter}'):
            f = os.path.join(f'extract/{letter}', image_name)
            if os.path.isfile(f):
                image = read_image(f'{f}')
                binarized_image = binarize_image(image, 1)
                inverted_image = invert_image(binarized_image)
                x_train.append(flatten_list(inverted_image))
                y_train.append(letter)
    print("Finished training with ETL dataset . . .")'''

    #if DEBUG and 0 == 1:
        # Write some images out just so we can see them visually.
     #   for idx, test_sample in enumerate(x_test):
      #      write_image(test_sample, f'{TEST_DIR}{idx}.png')
        #x_test = [read_image(f'{DATA_DIR}our_test.png')]
        #y_test = [7]

    '''print("----------------------------------------")

    output = open("output.txt", "a")

    for image_name in os.listdir('images/chars'):
        f = os.path.join('images/chars', image_name)
        print(image_name)
        if os.path.isfile(f):
            image = read_image(f'{f}')
            bw_im = binarize_image(image, 1)
            #for i in range(2):
             #   bw_im = thin_font(bw_im)
            inverted_image = invert_image(bw_im)
            cv2.imwrite("images/binarize_test.jpg", inverted_image)
            y_prediction = knn(x_train, y_train, [flatten_list(inverted_image)], 3)
            output.write(y_prediction[0])
            print(y_prediction[0])'''

    '''x_train = extract_features(x_train)
    x_test = extract_features(x_test)

    start = time.time()
    y_prediction = knn(x_train, y_train, x_test, 5)
    end = time.time()
    print(f'Prediction: {y_prediction}')
    print(f'Expected: {y_test}')
    #print(f'Prediction: {[chr(item + 64) for item in y_prediction]}')

    #print(f'Expected: {[chr(item + 64) for item in y_test]}')
    accuracy = sum([int(y_prediction_i == y_test_i)
                    for y_prediction_i, y_test_i in zip(y_prediction, y_test)]) / len(y_test)

    print(f'Accuracy: {accuracy * 100}%')'''

    start = time.time()
    neural_network_classifier(x_train, y_train, x_test, y_test, 3)
    end = time.time()
    #print(f'Elapsed time: {end - start}')


def extract_data():
    import struct

    location = DATA_DIR + "ETL1/ETL1C_04"
    filename = "ETL1C_04"

    RECORD_SIZE = 2052
    i = 0
    print("Reading {}".format(location))
    with open(location, 'rb') as f:
        while True:
            s = f.read(RECORD_SIZE)
            if s is None or len(s) < RECORD_SIZE:
                break
            r = struct.unpack(">H2sHBBBBBBIHHHHBBBBHH2016s4x", s)
            img = Image.frombytes('F', (64, 63), r[20], 'bit', (4, 0))
            img = img.convert('L')
            img = img.point(lambda x: 255 - (x << 4))
            i = i + 1
            dirname = r[1].decode('utf-8')
            dirname = dirname.replace('\0', '')
            dirname = dirname.replace(' ', '')
            dirname = dirname.replace('\\', 'YEN')
            dirname = dirname.replace('+', 'PLUS')
            dirname = dirname.replace('-', 'MINUS')
            dirname = dirname.replace('*', 'ASTERISK')
            try:
                os.makedirs(f"extract/{dirname}")
            except:
                pass
            imagefile = f"extract1/{dirname}/{filename}_{i:0>6}.png"
            print(imagefile)
            img.save(imagefile)
