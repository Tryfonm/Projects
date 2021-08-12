import numpy as np
from matplotlib import pyplot as plt


class NeuralNet():
    def __init__(self, input_layer=28 * 28, layer_1=20, output_layer=10):
        self.relu = np.vectorize(self._relu)
        self.drelu = np.vectorize(self._drelu)

        self.weights = {}
        self.Dweights = {}
        self.Dweights_accumulated = {}
        self.x = {}
        self.ksi = {}

        self.weights[1] = np.random.rand(input_layer + 1, layer_1) - 0.5
        self.weights[2] = np.random.rand(layer_1 + 1, output_layer) - 0.5

    def forward(self, input):

        self.batch_size, self.input_space = input.shape

        # Input layer
        self.x[0] = input
        self.x[0] = np.concatenate((self.x[0], np.ones((self.batch_size, 1))), axis=1)

        # Layer 1
        self.ksi[1] = self.x[0] @ self.weights[1]
        self.x[1] = self.relu(self.ksi[1])
        self.x[1] = np.concatenate((self.x[1], np.ones((self.batch_size, 1))), axis=1)

        # Layer 2
        self.ksi[2] = self.x[1] @ self.weights[2]
        self.x[2] = self.softmax(self.ksi[2])

        self.output = self.x[2]

        return self.output

    def backward(self, target):
        self.d = {}

        # Layer 2
        self.d[2] = (self.output - target)
        self.Dweights_accumulated[2] = self.x[1].T @ self.d[2]
        self.Dweights[2] = self.Dweights_accumulated[2] / self.batch_size  #####FIX

        # Layer 1
        self.d[1] = self.d[2] @ self.weights[2][0:self.weights[1].shape[1], :].T * self.drelu(self.ksi[1])

        self.Dweights_accumulated[1] = self.x[0].T @ self.d[1]
        self.Dweights[1] = self.Dweights_accumulated[1] / self.batch_size

    def step(self, lr=0.01):
        self.weights[1] = self.weights[1] - (lr * self.Dweights[1])
        self.weights[2] = self.weights[2] - (lr * self.Dweights[2])

    def softmax(self, x):
        x = x.T
        return (np.exp(x) / np.sum(np.exp(x), axis=0)).T

    @staticmethod
    def _relu(x):
        return np.maximum(0, x)

    @staticmethod
    def _drelu(x):
        return 1 if x > 0 else 0

    @staticmethod
    def accuracy(predictions_vector, true_labels_vector):
        return (predictions_vector == true_labels_vector).mean()

    @staticmethod
    def plot_acc(train_accuracies=None, test_accuracies=None, size=(10, 8), title=None):
        fig = plt.figure(figsize=(16, 8))
        if train_accuracies: plt.plot(train_accuracies, '.')
        if test_accuracies: plt.plot(test_accuracies, '.')
        if title: plt.title(title)
        plt.xlabel('Step')
        plt.ylabel('Accuracies')
        if train_accuracies and test_accuracies:
            plt.legend(['Train', 'Test'])
        elif train_accuracies:
            plt.legend(['Train'])
        else:
            plt.legend(['Test'])
