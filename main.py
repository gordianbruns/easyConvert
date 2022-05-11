from driver import run
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="Filename of the image")
    parser.add_argument("train_data", type=int, help="Number of training data")
    parser.add_argument("-t", "--type", type=int, help="Choose type of data (0: letters (default), 1: digits, 2: mixed)",
                        choices={0, 1, 2}, default=0)
    parser.add_argument("-d", "--debug", help="DEBUG mode on", action="store_true")
    parser.add_argument("-m", "--method", help="Uses k-nearest neighbor algorithm for OCR if activated "
                                               "(default: neural network)", action="store_true")
    parser.add_argument("-c", "--contours", help="Create image to show contours", action="store_true")
    parser.add_argument("-hi", "--histogram", help="Show histogram of pixel frequency", action="store_true")
    parser.add_argument("-f", "--filters", help="Show images after different steps of preprocessing",
                        action="store_true")
    parser.add_argument("-s", "--selection", help="Show a selection of different for binarization algorithms",
                        action="store_true")
    args = parser.parse_args()
    run(args)
