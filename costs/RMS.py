import numpy as np
import sys
import os

# HACK: I don't know how to get py files from folders in upper directory, copy
# pasted from stackoverflow.com
sys.path.append(os.path.dirname(__file__) + '/' + '../')

from activations.sigmoid import sigmoid_prime


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)
