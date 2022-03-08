import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import interpolation as inter


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


def processImage(input_image):
    print("processing image")
    image = cv2.imread(input_image)
    '''cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()'''
    display_image(input_image)
    invert_image(image)
    display_image("images/inverted.jpg")

    gray_image = grayscale(image)
    cv2.imwrite("images/gray.jpg", gray_image)

    bw_im = binarize_image(gray_image)
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

    cropped = remove_borders(no_noise)
    cv2.imwrite("images/cropped.jpg", cropped)
    display_image("images/cropped.jpg")

    new = cv2.imread("images/rotated.png")
    fixed = deskew(new)
    cv2.imwrite("images/rotated_fixed.jpg", fixed)
    display_image("images/rotated_fixed.jpg")

    image_with_border = add_borders(no_noise)
    cv2.imwrite("images/image_with_border.jpg", image_with_border)
    display_image("images/image_with_border.jpg")


    '''
    images1 = [image] + binarize_image(image)[0]
    images2 = [image] + binarize_image(image)[1]
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

    angle, rotated = correct_skew(image)
    print(angle)
    cv2.imshow('rotated', rotated)
    cv2.imwrite('images/rotated.png', rotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    color = cv2.cvtColor(images1[2], cv2.COLOR_BGR2BGRA)
    remove_noise(color)'''


# only needed for Tesseract 3.0
def invert_image(image):
    inverted_image = cv2.bitwise_not(image)
    cv2.imwrite("images/inverted.jpg", inverted_image)


def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def binarize_image(gray_image):
    thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)#cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite("images/bw_image.jpg", thresh)
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
    dilate = cv2.dilate(thresh, kernel, iterations=2)

    # Find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for c in contours:
        rect = cv2.boundingRect(c)
        x, y, w, h = rect
        cv2.rectangle(newImage, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Find largest contour and surround in min area box
    largestContour = contours[0]
    print(len(contours))
    minAreaRect = cv2.minAreaRect(largestContour)
    cv2.imwrite("temp/boxes.jpg", newImage)
    # Determine the angle. Convert it to the value that was originally used to obtain skewed image
    angle = minAreaRect[-1]
    if angle < -45:
        angle = 90 + angle
    print("Angle:", angle)
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
    cntsSorted = sorted(contours, key=lambda x:cv2.contourArea(x))
    cnt = cntsSorted[-1]
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y+h, x:x+w]
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

