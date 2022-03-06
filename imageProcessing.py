import cv2
import matplotlib.pyplot as plt


def processImage(input_image):
    print("processing image")
    image = read_image(input_image)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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


def read_image(input_image):
    image = cv2.imread(input_image, 0)
    return image


def binarize_image(image):
    ret, th1 = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    th4 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th5 = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)
    return [th1, th2, th3], [th2, th4, th5]

