import numpy
from typing import Tuple, List
from pandas import read_csv

Labels = List[str]


def _get_mnist_labels() -> Labels:
    """
    Creates labels for the mnist dataset attributes.

    :return: List of strings containing the labels.
    """
    # Create a list with the prediction label's name.
    names = ['number']

    # For every pixel, create a label containing the word 'pixel', followed by its index.
    for i in range(784):
        names.append('pixel' + str(i))

    return names


def _get_spam_labels() -> Labels:
    """
    Creates labels for the spambase dataset attributes.

    :return: List of strings containing the labels.
    """
    # Create an empty list.
    names = []

    # For all the word frequencies, create a label containing the word 'word-freq-', followed by its index.
    for i in range(48):
        names.append('word-freq-' + str(i))

    # For all the char frequencies, create a label containing the word 'char-freq-', followed by its index.
    for i in range(6):
        names.append('char-freq-' + str(i))

    # Create a list containing the remaining labels.
    labels = ['capital-run-length-average', 'capital-run-length-longest', 'capital-run-length-total', 'class']

    # Add remaining labels.
    for label in labels:
        names.append(label)

    return names


def load_digits(train: bool = True) -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Load the mnist handwritten digits dataset
    and convert the prediction labels to odd and even numbers.

    :param train: whether to load the train or the test data.
    If True, returns the train.

    If False, returns the test.

    Default value: True

    :return: Array containing the mnist handwritten digits dataset.
    """
    # Create a filename based on the train value.
    filename = 'datasets/mnist_train.csv' if train else 'datasets/mnist_test.csv'

    # Read the dataset and get its values.
    dataset = read_csv(filename, names=_get_mnist_labels()).values

    # Get x and y.
    x = dataset[:, 1:]
    y = dataset[:, 0]

    # Convert the prediction labels to odd and even numbers.
    y[y % 2 == 0] = 0
    y[y % 2 == 1] = 1

    return x, y


def load_spam() -> numpy.ndarray:
    """
    Loads the spambase dataset.

    :return: Numpy representation of the dataset.
    """
    filename = 'datasets/spambase.csv'
    return read_csv(filename, names=_get_spam_labels()).values


def get_email_name(class_num: int) -> str:
    """
    Gets an email's class name by its number.

    :param class_num: the number of the class name to be returned.
    :return: String containing the class name.
    """
    class_names = {
        0: 'Not Spam',
        1: 'Spam'
    }
    return class_names.get(class_num, 'Invalid')


def get_digit_name(class_num: int) -> str:
    """
    Gets a digit's class name by its number.

    :param class_num: the number of the class name to be returned.
    :return: String containing the class name.
    """
    class_names = {
        0: 'Even',
        1: 'Odd'
    }
    return class_names.get(class_num, 'Invalid')
