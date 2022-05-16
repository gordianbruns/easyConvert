README
--------------
Gordian Bruns

CSC 498B: Senior Project

EasyConvert: Convert Handwritten to Typed Text
--------------
I. File List
 - driver.py # contains run, extract_data
 - imageProcessing.py # contains display_image, process_image, invert_image, grayscale, binarize_image, noise_removal, thin_font, thick_font, get_skew_angle, rotate_image, deskew, remove_borders, add_borders, binarize_image_selection, read_images, read_labels, get_contours
 - main.py # calls driver.py
 - ocrAlgorithms.py # contains knn, neural_network_classifier
 - helpers (dir)
   - algorithmHelpers.py # contains dist, get_training_distances, get_most_frequent
   - common.py # contains bytes_to_int, bytes_to_char
   - imageProcessingHelpers.py # contains simplify_list, extract_features

II. Additional Folders
 - data # contains image files for training
 - images # used to store images
 - model # used to store the neural network model
 - test # used during debugging to store show some sample images

III. Usage

Before running the program, the requirements must be installed first.  
That can be done using the following command:  

pip install -r requirements.txt

EasyConvert assumes that python 3.X is installed.

The program takes two mandatory command line arguments:
 - filename
 - train_data_num

Depending on which algorithm is selected, slightly different things happen:
 - K-nearest neighbor algorithm: It processes the image and interprets the image to convert the writing in it to typed text. After that, the program runs a test of the algorithm with 100 images to check the accuracy.
 - Neural Network: It processes the image the same way but does not interpret the writing in it. The program then trains the neural network, and it will show the accuracy of the model.
In general, if only digits are selected for training, then the program will not interpret the image.

For example, if you want to run the program on the hello world example that is provided, using the k-nearest neighbor algorithm with 5,000 training samples, then you must enter the following into the command line:  

python main.py -m images/hello_world.jpg 5000


The general usage is:

python main.py [-h] [-t {0,1,2}] [-d] [-m] [-c] [-hi] [-f] [-s] filename train_data_num

positional arguments:
 - filename # Filename of the image 
 - train_data_num # Number of training data to use

optional arguments:
 - -h, --help # show this help message and exit
 - -t {0,1,2}, --type {0,1,2} # Choose type of data (0: letters (default), 1: digits, 2: mixed)
 - -d, --debug # DEBUG mode on
 - -m, --method # Uses k-nearest neighbor algorithm for OCR if activated (default: neural network)
 - -c, --contours # Create image to show contours
 - -hi, --histogram # Show histogram of pixel frequency
 - -f, --filters # Show images after different steps of preprocessing
 - -s, --selection # Show a selection of different for binarization algorithms

Note that you must enter a positive number for the number of training data.  
The sizes of the datasets are as follows:
 - letters: 145,600 characters
 - digits: 280,000 characters