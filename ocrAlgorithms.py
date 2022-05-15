from helpers.algorithmHelpers import *


# k-nearest neighbor algorithm
def knn(x_train: list, y_train: np.ndarray, x_test: list, k: int = 3) -> list:
    y_prediction = []  # store predictions
    count = 0  # used to keep track of how many characters have been classified

    # compare every test sample to our training data
    for idx, test_sample in enumerate(x_test):
        # compute how different the test_sample is from each other character
        training_distances = get_training_distances(test_sample, x_train)
        # sort by distance
        sorted_distance_indices = [
            pair[0] for pair in sorted(
                enumerate(training_distances),
                key=lambda x: x[1])
        ]

        # find k nearest neighbors
        neighbors = [y_train[index] for index in sorted_distance_indices[:k]]

        count += 1
        print(f'Classified {count}/{len(x_test)}')

        # get most frequent classification in neighbors
        nearest_neighbor = get_most_frequent(neighbors)

        # make prediction
        y_prediction.append(nearest_neighbor)

    return y_prediction


# algorithm using a neural network
def neural_network_classifier(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                              n_epochs: int = 3):
    n_hidden1 = 512  # 1st hidden layer
    n_hidden2 = 256  # 2nd hidden layer
    n_hidden3 = 128  # 3rd hidden layer
    n_output = 27  # output layer (0-9 digits or 26 letters)
    # -> letters are classified as 1-27; so we need 27 output nodes

    # flatten data
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # create neural network
    model = tf.keras.models.Sequential()
    # input layer with 28 * 28 nodes; this must match resolution of images
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))
    # 3 hidden layers
    model.add(tf.keras.layers.Dense(units=n_hidden1, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=n_hidden2, activation=tf.nn.relu))
    model.add(tf.keras.layers.Dense(units=n_hidden3, activation=tf.nn.relu))
    # output layer
    model.add(tf.keras.layers.Dense(units=n_output, activation=tf.nn.softmax))

    # transforms neural network into a series of matrix transformations
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # training the model
    model.fit(x_train, y_train, epochs=n_epochs)

    # measures performance of model
    loss, accuracy = model.evaluate(x_test, y_test)
    print("Accuracy:", accuracy)
    print("Loss:", loss)

    # save the model
    model.save('model')
