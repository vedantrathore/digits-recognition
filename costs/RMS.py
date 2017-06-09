import numpy as np
import sys
import os
sys.path.append(os.path.dirname(__file__) + '/' + '../')

from activations.sigmoid import sigmoid_prime


class QuadraticCost(object):

    @staticmethod
    def fn(a, y):
        """Return the cost associated with an output ``a`` and desired output
        ``y``.
        """
        return 0.5 * np.linalg.norm(a - y)**2

    @staticmethod
    def delta(z, a, y):
        """Return the error delta from the output layer."""
        return (a - y) * sigmoid_prime(z)
