from costs import RMS, CrossEntropyCost
from activations import sigmoid

# Dict of all the cost functions
costs = {
    'rms': RMS.QuadraticCost(),
    'cross_entropy': CrossEntropyCost.CrossEntropyCost()
}

# Dict of all the activation functions
activations = {
    'sigmoid': sigmoid,
}
