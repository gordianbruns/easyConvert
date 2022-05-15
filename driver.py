import argparse

from imageProcessing import *
from ocrAlgorithms import knn, neural_network_classifier
from emnist import extract_training_samples, extract_test_samples
import os
from PIL import Image
import time
import pytesseract

DEBUG = True
USE_TESSERACT = False
USE_ETL = False

if DEBUG:
    import numpy as np


    def read_image(path: str) -> np.ndarray:
        image = Image.open(path).convert('L')
        image = image.resize((32, 32))
        image = add_borders(image)
        return np.asarray(image)


    def write_image(image: Image, path: str):
        img = Image.fromarray(np.array(image), 'L')
        img.save(path)


ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

# directories
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


# main driver function
def run(args: argparse.Namespace):
    # processes image according to specified options
    process_image(args)

    method = args.method  # False = neural network; True = k-nearest neighbor
    mixed = False   # to indicate whether the digits' data set must be added after the letters

    # parse what data should be used
    training_data_type = ''
    if args.type == 0:
        training_data_type = 'letters'
    elif args.type == 1:
        training_data_type = 'digits'
    elif args.type == 2:
        training_data_type = 'letters'
        mixed = True

    print("Training with EMNIST dataset . . .")

    # extract training data
    x_train, y_train = extract_training_samples(training_data_type)

    # reduce training data to desired size
    # note that the neural network and the k-nearest neighbor algorithm take different data types
    x_train = x_train[:args.train_data_num]
    if method and training_data_type != 'digits':
        # convert numbers to ASCII characters
        y_train = [chr(char + 64) for char in y_train[:args.train_data_num]]
    else:
        y_train = y_train[:args.train_data_num]

    # add digits if mixed
    if mixed:
        new_x_train, new_y_train = extract_training_samples('digits')
        new_x_train, new_y_train = new_x_train[:args.train_data_num], new_y_train[:args.train_data_num]
        x_train = np.concatenate((x_train, new_x_train), axis=0)
        y_train = list(y_train) + list(new_y_train)

    # extract test data
    x_test, y_test = extract_test_samples(training_data_type)

    # convert y_test data if needed
    if method and training_data_type != 'digits':
        y_test = [chr(char + 64) for char in y_test[:args.train_data_num]]

    # add digits if mixed
    if mixed:
        new_x_test, new_y_test = extract_test_samples('digits')
        x_test = np.concatenate((x_test, new_x_test), axis=0)
        y_test = list(y_test) + list(new_y_test)

    # reduce test data to 100
    x_test = x_test[:len(x_test):int(len(x_test) / 100)]
    y_test = y_test[:len(y_test):int(len(y_test) / 100)]

    print("Finished training with EMNIST dataset . . .")

    # uses Google's tesseract OCR software
    if USE_TESSERACT:
        correct_predictions = 0
        # config for software
        config = r"--psm 10 --oem 3"

        start = time.time()

        # classify every image from the test data
        for image_index in range(len(x_test)):
            new_image = Image.fromarray(x_test[image_index])
            prediction = pytesseract.image_to_string(new_image, config=config)
            print(prediction)
            if prediction == y_test[image_index]:
                correct_predictions += 1

        end = time.time()

        print(f'Accuracy: {(correct_predictions/len(x_test)) * 100}%')

    # trains with the ETL database in addition
    if USE_ETL:
        print("Starting training with ETL dataset . . .")

        # convert and preprocess images
        y_train = y_train.astype(str)
        for image in list(x_train):
            new_image = Image.fromarray(image)
            new_image = add_borders(new_image)
            new_image = np.asarray(new_image)
            binarized_image = binarize_image(new_image, 1)
            inverted_image = invert_image(binarized_image)
            x_train.append(simplify_list(list(inverted_image)))

        # train with every letter
        for letter in ALPHABET:
            print(f"Training {letter} . . .")
            for image_name in os.listdir(f'extract/{letter}'):
                f = os.path.join(f'extract/{letter}', image_name)
                if os.path.isfile(f):
                    image = read_image(f'{f}')
                    binarized_image = binarize_image(image, 1)
                    inverted_image = invert_image(binarized_image)
                    x_train.append(simplify_list(list(inverted_image)))
                    y_train.append(letter)
        print("Finished training with ETL dataset . . .")

    if args.debug:
        # write some images out just so we can see them visually
        for idx, test_sample in enumerate(x_test):
            write_image(test_sample, f'{TEST_DIR}{idx}.png')

    print("----------------------------------------")

    # if k-nearest neighbor algorithm is used
    if method:

        x_train = extract_features(x_train)

        # only interpret given image if we trained with letters
        if training_data_type == 'letters' or training_data_type == 'mixed':
            output = open("output.txt", "a")
            print("Start image interpretation . . .")
            print()

            # interpret every image of the segmented characters
            for image_name in os.listdir('images/chars'):
                f = os.path.join('images/chars', image_name)
                print(image_name)
                if os.path.isfile(f):
                    # get image
                    image = read_image(f'{f}')

                    # preprocess image
                    bw_im = binarize_image(image, 1)
                    # for i in range(2):    # sometimes necessary to thin it
                    #    bw_im = thin_font(bw_im)
                    inverted_image = invert_image(bw_im)

                    # show what the image looks like when it is being interpreted
                    cv2.imwrite("images/binarize_test.jpg", inverted_image)

                    # get prediction
                    y_prediction = knn(x_train, y_train, [simplify_list(list(inverted_image))], 3)

                    # write it to the file and print it out
                    output.write(str(y_prediction[0]))
                    print(y_prediction[0])

            print()
            print("Image interpretation completed")
            print()

        # run the k-nearest neighbor algorithm on dataset samples
        start = time.time()
        y_prediction = knn(x_train, y_train, x_test, 5)
        end = time.time()
        print(f'Prediction: {y_prediction}')
        print(f'Expected: {y_test}')

        accuracy = sum([int(y_prediction_i == y_test_i)
                        for y_prediction_i, y_test_i in zip(y_prediction, y_test)]) / len(y_test)

        print(f'Accuracy: {accuracy * 100}%')

    # if neural network
    else:
        # run the neural network on dataset samples
        start = time.time()
        neural_network_classifier(x_train, np.asarray(y_train), x_test, np.asarray(y_test), 3)
        end = time.time()

    print(f'Elapsed time: {end - start} seconds')


# function to extract data from ETL database
def extract_data():
    import struct

    location = DATA_DIR + "ETL1/ETL1C_04"
    filename = "ETL1C_04"

    RECORD_SIZE = 2052      # size of one image: 64 x 63
    i = 0
    print("Reading {}".format(location))
    with open(location, 'rb') as f:
        while True:
            # read record
            s = f.read(RECORD_SIZE)
            if s is None or len(s) < RECORD_SIZE:
                break

            # decrypt
            r = struct.unpack(">H2sHBBBBBBIHHHHBBBBHH2016s4x", s)

            # read in image
            img = Image.frombytes('F', (64, 63), r[20], 'bit', (4, 0))
            img = img.convert('L')
            img = img.point(lambda x: 255 - (x << 4))
            i = i + 1

            # save image in respective directory
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
