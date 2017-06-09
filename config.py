from costs import RMS, CrossEntropyCost
from activations import sigmoid

costs = {
    'rms': RMS.QuadraticCost(),
    'cross_entropy': CrossEntropyCost.CrossEntropyCost()
}

activations = {
    'sigmoid': sigmoid,
}
