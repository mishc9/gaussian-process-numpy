import numpy as np


# Define the kernel function
def kernel(a, b, param):
    sqdist = np.sum(a ** 2, 1).reshape(-1, 1) + np.sum(b ** 2, 1) - 2 * np.dot(a, b.T)
    return np.exp(-0.5 * (1 / param) * sqdist)


def get_train_data(train_points, fn):
    """
    So, we'll try to reconstruct a sine function with GP
    :param train_points:
    :return:
    """
    x_train = np.array(train_points)  # .reshape(5, 1)
    train_size = len(x_train)
    x_train = x_train.reshape(train_size, 1)
    y_train = fn(x_train)
    return x_train, y_train