import os
import random
import numpy as np

from activations.sigmoid import sigmoid, sigmoid_prime
from datasets import mnist
from config import costs


class Network(object):
    """Initialize a Neural Network model

    Parameters
    ---------
    sizes: list, optional
        A list of numbers specifying number of neurons in each layer. Not
        required if pretrained model is used.

    learning_rate: float, optional
        learning rate for the gradient descent optimization. Defaults to 3.0

    batch_size: int, optional
        Size of the mini batch of training examples as used by Stochastic
        Gradient Descent. Denotes after how many examples the weights and biases
        would be updated. Default size is 10.

    """

    def __init__(self, sizes=list(), epochs=10, learning_rate=3.0,
                 batch_size=10, cost='rms'):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes]
        self.cost = costs[cost]
        # first term is for layer 0 (input layer)
        self.weights = [np.array([0])] + [np.random.randn(y, x)
                                          for y, x in zip(sizes[1:], sizes[:-1])]
        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.batch_size = batch_size
        self.epochs = epochs
        self.eta = learning_rate

    def fit(self, training_data, validation_data=None):
        """Fit(train) the Neural Network on provided training data. Fitting is
        carried out using Stochastic Gradient Descent Algorithm.

        Parameters
        ---------
        training_data : list of tuple
            A list of tuples of numpy arrays, ordered as (image, label).

        validation_data : list of tuple, optional
            Same as `training_data`, if provided, the netowrk will display the
            validation accuracy after each epoch

        """
        for epoch in range(self.epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k + self.batch_size] for k in
                range(0, len(training_data), self.batch_size)]

            for mini_batch in mini_batches:
                nabla_b = [np.zeros(b.shape) for b in self.biases]
                nabla_w = [np.zeros(w.shape) for w in self.weights]
                for x, y in mini_batch:
                    self._forward_prop(x)
                    delta_nabla_b, delta_nabla_w = self._back_prop(x, y)
                    nabla_b = [nb + dnb for nb,
                               dnb in zip(nabla_b, delta_nabla_b)]
                    nabla_w = [nw + dnw for nw,
                               dnw in zip(nabla_w, delta_nabla_w)]

                self.weights = [
                    w - (self.eta / self.batch_size) * dw for w, dw in
                    zip(self.weights, nabla_w)]

                self.biases = [
                    b - (self.eta / self.batch_size) * db for b, db in
                    zip(self.biases, nabla_b)]

            if validation_data:
                accuracy = self.validate(validation_data) / 100.00
                print "Epoch {0}: accuracy {1} %.".format(epoch + 1, accuracy)
            else:
                print "Epoch {0} complete".format(epoch + 1)

    def validate(self, validation_data):
        """Validate the Neural Netwoek on provided validation data. It uses the
        number of correctly predicted examples as validation accuracy metric.

        Parameters
        --------
        validation_data : list of tuple

        Returns
        ------
        int
            Number of correctly predicted images
        """
        validation_results = [(self.predict(x) == y)
                              for x, y in validation_data]
        return sum(result for result in validation_results)

    def predict(self, x):
        """Predict the label of a single test example (image).

        Parameters
        ---------
        x : numpy.array

        Returns
        ------
        int
            Predicted label of example (image)
        """
        self._forward_prop(x)
        return np.argmax(self._activations[-1])

    def _forward_prop(self, x):
        self._activations[0] = x
        for i in range(1, self.num_layers):
            self._zs[i] = (
                self.weights[i].dot(self._activations[i - 1]) + self.biases[i]
            )
            self._activations[i] = sigmoid(self._zs[i])

    def _back_prop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        error = (self.cost).delta(self._zs[-1], self._activations[-1], y)

        nabla_b[-1] = error
        nabla_w[-1] = error.dot(self._activations[-2].transpose())

        for l in range(self.num_layers - 2, 0, -1):
            error = np.multiply(
                self.weights[l + 1].transpose().dot(error),
                sigmoid_prime(self._zs[l])
            )
            nabla_b[l] = error
            nabla_w[l] = error.dot(self._activations[l - 1].transpose())

        return nabla_b, nabla_w

    def load(self, filename='model.npz'):
        """Prepare a neural network from a compressed binary containing weights and
        biases array. Size of layers are derived from dimensions of numpy arrays.

        Parameters
        ---------
        filename : str, optional
            Name of the ``.npz`` compressed binary in models directory.


        """
        npz_members = np.load(os.path.join(os.curdir, 'models', filename))

        self.weights = list(npz_members['weights'])
        self.biases = list(npz_members['biases'])
        self.sizes = [b.shape[0] for b in self.biases]
        self.num_layers = len(self.sizes)
        self.cost = npz_members['cost']

        self._zs = [np.zeros(bias.shape) for bias in self.biases]
        self._activations = [np.zeros(bias.shape) for bias in self.biases]

        self.epochs = int(npz_members['epochs'])
        self.batch_size = int(npz_members['batch_size'])
        self.epochs = int(npz_members['epochs'])

    def save(self, filename='model.npz'):
        """Save weights, biases and hyperparameters of a neural network to a
        compressed binary. This ``.npz`` binary is saved in 'models' directory

        Parameters
        ---------
        filename: str, optional
            Name of the ``.npz`` compressed binary in to be saved


        """
        if not os.path.exists('models'):
            os.makedirs('models')
        np.savez_compressed(
            file=os.path.join(os.curdir, 'models', filename),
            weights=self.weights,
            biases=self.biases,
            batch_size=self.batch_size,
            epochs=self.epochs,
            eta=self.eta,
            cost=self.cost,
        )


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist.load_data()
    nn = Network(sizes=[784, 30, 10], cost='cross_entropy')
    print "--------------- Training ---------------"
    nn.fit(training_data=training_data, validation_data=validation_data)
    print "Neural Network accuracy on test data is {} %".format(nn.validate(test_data) / 100.00)
    nn.save()
    print " -------------- Complete ---------------"
