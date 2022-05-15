import argparse

from helpers.imageProcessingHelpers import *


# debug function: shows image at specific path
def display_image(image_path: str):
    dpi = 80    # dots per inch; normalize size

    # read in image
    try:
        image_data = plt.imread(image_path)
    except FileNotFoundError:
        warnings.warn(f'Image at path {image_path} not found')
        return

    # get dimensions of image
    if len(image_data.shape) == 3:
        height, width, depth = image_data.shape
    else:
        height, width = image_data.shape

    # normalize image
    figsize = width / float(dpi), height / float(dpi)

    # create plot to show image
    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])

    ax.axis('off')

    ax.imshow(image_data, cmap='gray')

    plt.show()


# main image processing function
def process_image(args: argparse.Namespace):
    print("processing image. . .")

    # initialize variables based on parameters
    input_image = args.filename
    SHOW_CONTOURS = args.contours
    SHOW_HISTOGRAMS = args.histogram
    SHOW_IMAGES = args.filters
    SHOW_SELECTION = args.selection

    # read in image
    image = cv2.imread(input_image)
    if image is None:
        raise FileNotFoundError(f'Image at path {input_image} not found.')

    # if it should create an image that shows the contours
    if SHOW_CONTOURS:
        get_contours(image)

    # if it should show a histogram that shows how many black and white pixels are in the image horizontally
    if SHOW_HISTOGRAMS:
        # preprocessing
        inverted = invert_image(image)
        gray_image = grayscale(inverted)
        bw_im = binarize_image(gray_image, 1)

        no_noise = noise_removal(bw_im)

        hist = []
        indices = [i for i in range(len(no_noise))]

        # create histogram
        for item in no_noise:
            normalized_item = [x / 255 for x in item]
            hist.append(sum(normalized_item))

        # show histogram
        plt.barh(indices, hist)
        plt.show()

    # show different images of the preprocessing steps
    if SHOW_IMAGES:
        # show original image
        display_image(input_image)
        # show inverted image
        invert_image(image)
        display_image("images/inverted.jpg")

        # show the effect of grayscaling
        gray_image = grayscale(image)
        cv2.imwrite("images/gray.jpg", gray_image)

        # show the effect of binarization
        bw_im = binarize_image(gray_image, 0)
        display_image("images/bw_image.jpg")

        # show the effect of noise removal
        no_noise = noise_removal(bw_im)
        cv2.imwrite("images/no_noise.jpg", no_noise)
        display_image("images/no_noise.jpg")

        # show the effect of making the font thinner and thicker
        thin_image = thin_font(no_noise)
        thick_image = thick_font(no_noise)
        cv2.imwrite("images/eroded_image.jpg", thin_image)
        cv2.imwrite("images/dilated_image.jpg", thick_image)
        display_image("images/eroded_image.jpg")
        display_image("images/dilated_image.jpg")

        # show original rotated image
        new = cv2.imread("images/example_rotated.png")
        display_image("images/example_rotated.png")
        # deskew image and show it
        fixed = deskew(new)
        cv2.imwrite("images/rotated_fixed.jpg", fixed)
        display_image("images/rotated_fixed.jpg")

    # show a selection of different binarization parameters and algorithms
    if SHOW_SELECTION:
        # images
        images1 = [image] + binarize_image_selection(grayscale(image))[0]
        images2 = [image] + binarize_image_selection(grayscale(image))[1]

        # labels
        titles1 = ['Original Image', 'Global Thresholding (v = 127)', 'Adaptive Mean Thresholding',
                   'Adaptive Mean Thresholding (10)']
        titles2 = ['Original Image', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding',
                   'Adaptive Gaussian Thresholding(10)']

        # show the first four images
        for i in range(len(images1)):
            plt.subplot(2, 2, i + 1), plt.imshow(images1[i], 'gray')
            plt.title(titles1[i])
            plt.xticks([]), plt.yticks([])
        plt.show()

        # show the second four images
        for i in range(len(images2)):
            plt.subplot(2, 2, i + 1), plt.imshow(images2[i], 'gray')
            plt.title(titles2[i])
            plt.xticks([]), plt.yticks([])
        plt.show()


# function to invert colors of the image
def invert_image(image: np.ndarray) -> np.ndarray:
    inverted_image = cv2.bitwise_not(image)
    return inverted_image


# function to convert the image to grayscale
def grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# function to binarize image
def binarize_image(gray_image: np.ndarray, method: int) -> np.ndarray:
    # method 0 uses an adaptive algorithm while method 1 uses a static algorithm

    if method == 0:
        thresh = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    else:
        ret, thresh = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
    return thresh


# function to remove noise from an image
def noise_removal(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((1, 1), np.uint8)  # needed for image processing

    # increase area of characters
    image = cv2.dilate(image, kernel, iterations=1)
    # erode boundaries of characters
    image = cv2.erode(image, kernel, iterations=1)
    # closes small dots within the characters
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # blurs the image slightly
    image = cv2.medianBlur(image, 3)

    return image


# function to thin out the characters
def thin_font(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)  # needed for image processing

    # invert image
    image = invert_image(image)
    # erode boundaries of characters
    image = cv2.erode(image, kernel, iterations=1)
    # invert image back to original
    image = invert_image(image)

    return image


# function to make the characters thicker
def thick_font(image: np.ndarray) -> np.ndarray:
    kernel = np.ones((2, 2), np.uint8)  # needed for image processing

    # invert image
    image = invert_image(image)
    # increase area of characters
    image = cv2.dilate(image, kernel, iterations=1)
    # invert image back to original
    image = invert_image(image)

    return image


# function to calculate skew angle of an image
def get_skew_angle(image: np.ndarray) -> float:
    # prepare image, copy, convert to gray scale, blur, and threshold
    new_image = image.copy()
    gray = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # apply dilate to merge text into meaningful lines/paragraphs.
    # use larger kernel on X axis to merge characters into single line, cancelling out any spaces.
    # but use smaller kernel on Y axis to separate between different blocks of text
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 5))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # find all contours
    contours, hierarchy = cv2.findContours(dilate, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # find largest contour and surround in min area box
    largest_contour = contours[0]
    min_area_rect = cv2.minAreaRect(largest_contour)

    # determine angle and convert it to the value that was originally used to obtain skewed image
    angle = min_area_rect[-1]
    if angle < -45:
        angle = 90 + angle

    return -1.0 * angle


# function to rotate the image around its center
def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    # copy image
    new_image = image.copy()
    # get height and width of image
    (h, w) = new_image.shape[:2]
    # compute coordinates of center
    center = (w // 2, h // 2)

    # get rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    # create new image
    new_image = cv2.warpAffine(new_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    return new_image


# wrapper function to deskew image
def deskew(image: np.ndarray) -> np.ndarray:
    angle = get_skew_angle(image)

    return rotate_image(image, -1.0 * angle)


# function to remove borders around paper
def remove_borders(image: np.ndarray) -> np.ndarray:
    # get overall paper borders and structure
    contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x: cv2.contourArea(x))
    cnt = contours[-1]

    # create new image where everything outside the borders is cropped
    x, y, w, h = cv2.boundingRect(cnt)
    crop = image[y:y + h, x:x + w]

    return crop


# function to add borders around paper
def add_borders(image: Image) -> Image:
    old_size = image.size   # remember old size

    # create new image with normalized size
    new_size = (28, 28)
    new_im = Image.new(mode="RGB", size=new_size, color="white")
    # paste image into the middle of the new image
    new_im.paste(image, ((new_size[0] - old_size[0]) // 2, (new_size[1] - old_size[1]) // 2))

    return new_im


# function to try different binarization algorithms
def binarize_image_selection(gray_image: np.ndarray) -> tuple[list, list]:
    ret, th1 = cv2.threshold(gray_image, 210, 230, cv2.THRESH_BINARY)
    th2 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    th3 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 10)
    th4 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    th5 = cv2.adaptiveThreshold(gray_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 10)

    return [th1, th2, th3], [th2, th4, th5]


# deprecated since EMNIST has their own extract_training_samples
def read_images(filename: str, n_max_images: int = None) -> list:
    
    warnings.warn('read_images is deprecated', DeprecationWarning, stacklevel=2)
    
    images = []     # to store the images
    
    with open(filename, 'rb') as f:
        _ = f.read(4)  # type of file; can be ignored since known
        num_images = bytes_to_int(f.read(4))
        # reduce to max num of images

        if n_max_images:
            num_images = n_max_images
        num_rows = bytes_to_int(f.read(4))
        num_columns = bytes_to_int(f.read(4))
        
        # read image
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


# deprecated since EMNIST has their own extract_training_samples
def read_labels(filename: str, is_digits: bool, n_max_labels: int = None) -> list:
    
    warnings.warn('read_labels is deprecated', DeprecationWarning, stacklevel=2)
    
    labels = []     # to store the labels
    
    with open(filename, 'rb') as f:
        _ = f.read(4)  # type of file; can be ignored since known
        num_labels = bytes_to_int(f.read(4))
        # reduce to max num of labels
        if n_max_labels:
            num_labels = n_max_labels
        
        # read label
        for label_index in range(num_labels):
            if is_digits:
                label = bytes_to_int(f.read(1))
            else:
                label = bytes_to_char(f.read(1))
            labels.append(label)

    return labels


# function to find the contours of each letter and to create a new image for each letter
def get_contours(image: np.ndarray):
    # preprocessing
    gray = grayscale(image)
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    bw_im = binarize_image(blur, 1)
    thresh = cv2.threshold(bw_im, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 13))
    dilate = cv2.dilate(thresh, kernel, iterations=5)

    # extract contours
    contours = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
    contours = contours[0] if len(contours) == 2 else contours[1]
    # sort from left to right
    contours = sorted(contours, key=lambda x: cv2.boundingRect(x)[0])

    # create image for each letter
    for index in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[index])
        if w > 8 or h > 8:
            roi = image[y:y + h, x:x + w]
            cv2.imwrite(f'images/chars/index_roi{index}.png', roi)
            cv2.rectangle(image, (x, y), (x + w, y + h), (36, 255, 12), 2)
    # show how the image was segmented
    cv2.imwrite("images/index_bbox_new1.png", image)
