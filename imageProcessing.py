import cv2
import matplotlib.pyplot as plt
import numpy as np
from helpers.common import *

SHOW_CONTOURS = True
SHOW_HISTOGRAMS = True
SHOW_IMAGES = False
SHOW_SELECTION = False


def display_image(image_path):
    dpi = 80
    image_data = plt.imread(image_path)
    if len(image_data.shape) == 3:
        height, width, depth = image_data.shape
    else:
        height, width = image_data.shape

    figsize = width / float(dpi), height / float(dpi)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(image_data, cmap='gray')

    plt.show()


def process_image(input_image):
    print("processing image. . .")
    image = cv2.imread(input_image)

    if SHOW_CONTOURS:
        get_contours(image)

    if SHOW_HISTOGRAMS:
        inverted = invert_image(image)
        gray_image = grayscale(inverted)
        bw_im = binarize_image(gray_image, 1)

        no_noise = noise_removal(bw_im)

        hist = []
        indices = [i for i in range(len(no_noise))]

        for item in no_noise:
            normalized_item = [x / 255 for x in item]
            hist.append(sum(normalized_item))

        plt.barh(indices, hist)
        plt.show()

    if SHOW_IMAGES:
        display_image(input_image)
        invert_image(image)
        display_image("images/inverted.jpg")

        gray_image = grayscale(image)
        cv2.imwrite("images/gray.jpg", gray_image)

        bw_im = binarize_image(gray_image, 0)
        display_image("images/bw_image.jpg")

        no_noise = noise_removal(bw_im)
        cv2.imwrite("images/no_noise.jpg", no_noise)
        display_image("images/no_noise.jpg")

        thin_image = thin_font(no_noise)
        thick_image = thick_font(no_noise)
        cv2.imwrite("images/eroded_image.jpg", thin_image)
        cv2.imwrite("images/dilated_image.jpg", thick_image)
        display_image("images/eroded_image.jpg")
        display_image("images/dilated_image.jpg")

        # with_borders = cv2.imread("images/with_borders.jpg")
        # gray = grayscale(with_borders)
        # bw_im2 = binarize_image(gray, 1)
        # no_noise2 = noise_removal(bw_im2)
        # cropped = remove_borders(no_noise2)
        # cv2.imwrite("images/cropped.jpg", cropped)
        # display_image("images/cropped.jpg")

        new = cv2.imread("images/example_rotated.png")
        display_image("images/example_rotated.png")
        fixed = deskew(new)
        cv2.imwrite("images/rotated_fixed.jpg", fixed)
        display_image("images/rotated_fixed.jpg")

        image_with_border = add_borders(no_noise)
        cv2.imwrite("images/image_with_border.jpg", image_with_border)
        display_image("images/image_with_border.jpg")

    if SHOW_SELECTION:
        images1 = [image] + binarize_image_selection(grayscale(image))[0]
        images2 = [image] + binarize_image_selection(grayscale(image))[1]
        titles1 = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding',
                   'Adaptive Mean Thresholding (10)']
        titles2 = ['Original Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
                   'Adaptive Gaussian Thresholding(10)']
        for i in range(len(images1)):
            plt.subplot(2, 2, i + 1), plt.imshow(images1[i], 'gray')
            plt.title(titles1[i])
            plt.xticks([]), plt.yticks([])
        plt.show()
        for i in range(len(images2)):
            plt.subplot(2, 2, i + 1), plt.imshow(images2[i], 'gray')
            plt.title(titles2[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


# only needed for Tesseract 3.0
def invert_image(image):
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_image(gray_image, method):
    if method == 0:
        thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return thresh


def noise_removal(image):
    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    image = cv2.medianBlur(image, 3)
    return image


def thin_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.erode(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


def thick_font(image):
    image = cv2.bitwise_not(image)
    kernel = np.ones((2, 2), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.bitwise_not(image)
    return image


# Calculate skew angle of an image
def getSkewAngle(cvImage) -> float:
    # Prep image, copy, convert to gray scale, blur, and threshold
    newImage = cvImage.copy()
    gray = cv2.cvtColor(newImage, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Apply dilate to merge text into meaningful lines/paragraphs.
    # Use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # But use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    minAreaRect = cv2.minAreaRect(largestContour)

    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    return -1.0 * angle


# Rotate the image around its center
def rotateImage(cvImage, angle: float):
    newImage = cvImage.copy()
    (h, w) = newImage.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    newImage = cv2.warpAffine(newImage, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return newImage


# Deskew image
def deskew(cvImage):
    angle = getSkewAngle(cvImage)
    return rotateImage(cvImage, -1.0 * angle)


def remove_borders(image):
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cntsSorted = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]
    return crop


def add_borders(image):
    color = [0, 0, 0]
    top, bottom, left, right = [150] * 4
    image_with_border = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image_with_border


def binarize_image_selection(gray_image):
    ret, th1 = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    th4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th5 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    return [th1, th2, th3], [th2, th4, th5]


# helper function so that we have a one dimensional list
def flatten_list(l):
    return [pixel for element in l for pixel in element]


# helper function to extract features
def extract_features(x_data):
    return [flatten_list(sample) for sample in x_data]


def read_images(filename, n_max_images=None):
    images = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # type of file; can be ignored since known
        num_images = bytes_to_int(f.read(4))
        if n_max_images:
            num_images = n_max_images
        num_rows = bytes_to_int(f.read(4))
        num_columns = bytes_to_int(f.read(4))
        for img_index in range(num_images):
            image = []
            for row_index in range(num_rows):
                row = []
                for col_index in range(num_columns):
                    pixel = f.read(1)
                    row.append(pixel)
                image.append(row)
            images.append(image)
    return images


def read_labels(filename, n_max_labels=None):
    labels = []
    with open(filename, 'rb') as f:
        _ = f.read(4)  # type of file; can be ignored since known
        num_labels = bytes_to_int(f.read(4))
        if n_max_labels:
            num_labels = n_max_labels
        for label_index in range(num_labels):
            label = bytes_to_int(f.read(1))
            labels.append(label)
    return labels


def get_contours(image):
    gray = grayscale(image)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts = sorted(cnts, key=lambda x: cv2.boundingRect(x)[0])
    for index in range(len(cnts)):
        x, y, w, h = cv2.boundingRect(cnts[index])
        roi = image[y:y + h, x:x + w]
        cv2.imwrite(f'images/chars/index_roi{index}.png', roi)
        cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    cv2.imwrite("images/index_bbox_new.png", image)
