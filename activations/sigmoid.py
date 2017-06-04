import numpy as np


def sigmoid(x):
    """Sigmoid activation function"""
    return 1 / (1 + np.exp(-x))


def sigmoid_prime(x):
    """derivative of sigmoid function"""
    return sigmoid(x) * (1 - sigmoid(x))
