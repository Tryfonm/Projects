import numpy as np
from matplotlib import pyplot as plt


class NeuralNet():
    """
    A simple Neural Network implementation for classification purposes using numpy. Hidden layer activations are all ReLU's and the final layer is a softmax.
    """

    def __init__(self, input_layer: int, hidden_layers: list, output_layer: int):
        """
        Neural network initialization method tha expects the input space (number of total features of the dataset), the number of neurons for each hidden layer (as a list) and the output dimension (number of total classes)

        Parameters
        ----------
        input_layer: [int] Input space of the neural network
        hidden_layers: [list] A list of integers, each corresponding to the number of neurons. For example [100 50 25] creates 3 hidden layers with 100, 50 and 25 neurons each
        output_layer: [int] Output dimension. Should be exactly similar to the target_vectors used in backward method.

        """
        self.layers = [input_layer] + hidden_layers + [output_layer]
        self.final_layer_index = len(self.layers) - 1

        self.relu = np.vectorize(self._relu)
        self.drelu = np.vectorize(self._drelu)

        self.weights = {}
        self.d = {}
        self.Dweights = {}
        self.x = {}
        self.ksi = {}

        # Weight initialization
        for layer in range(self.final_layer_index):
            self.weights[layer + 1] = np.random.rand(self.layers[layer] + 1, self.layers[layer + 1]) - 0.5

    def forward(self, input_vectors: np.ndarray):
        """
        Feedforward the input vector (if fed with a single data-point) / matrix (if fed with a batch of data-points). The first dimension defines the batch_size, meaning that 'NeuralNet.backward' method expects the exact same number of batch_size (1st dimension).

        Parameters
        ----------
        input_vectors: [numpy.array] Expects a np.array of shape (batch_size, input_layer). The output layer is defined in 'NeuralNet.__init__' method.

        Returns
        -------
        A numpy.array of shape (batch_size, output_layer). The output layer is defined in 'NeuralNet.__init__' method.

        """
        self.batch_size, self.input_space = input_vectors.shape

        # Input layer
        self.x[0] = input_vectors.T
        ones = np.ones((1, self.batch_size))
        self.x[0] = np.concatenate((self.x[0], np.ones((1, self.batch_size))), axis=0)

        # Hidden Layers
        for layer in range(1, len(self.layers) - 1):
            # Layer 1
            self.ksi[layer] = self.weights[layer].T @ self.x[layer - 1]
            self.x[layer] = self.relu(self.ksi[layer])
            self.x[layer] = np.concatenate((self.x[layer], np.ones((1, self.batch_size))), axis=0)

        # Output Layer
        self.ksi[layer + 1] = self.weights[layer + 1].T @ self.x[layer]
        self.x[layer + 1] = self.softmax(self.ksi[layer + 1])

        self.output = self.x[layer + 1].T  # reshaping the output to look similar to tensors dimensions (batch, feature)
        return self.output

    def backward(self, target_vectors: np.ndarray):
        """
        Backpropagate through the network so as to find the gradients that minimize the loss (CrossEntropy loss) of a Softmax final layer.

        Parameters
        ----------
        target_vectors: [numpy.array] A numpy.array of shape (batch_size, output_layer) that represents the one-hot-encoded true labels of the dataset.

        """
        # Final layer
        self.d[self.final_layer_index] = (self.output - target_vectors).T  # Softmax combined with CrossEntropy loss boils down to this
        self.Dweights[self.final_layer_index] = self.x[self.final_layer_index - 1] @ self.d[self.final_layer_index].T / self.batch_size

        # Hidden layers
        for layer in range(self.final_layer_index, 1, -1):
            self.d[layer - 1] = self.weights[layer][0:self.weights[layer - 1].shape[1], :] @ self.d[layer] * self.drelu(self.ksi[layer - 1])
            self.Dweights[layer - 1] = self.x[layer - 2] @ self.d[layer - 1].T / self.batch_size

    def step(self, lr=0.01):
        """
        Updates the weight parameters towards the optimal direction.

        Parameters
        ----------
        lr: [flat] A scalar value controlling the stepsize that is made towards the optimal values.

        """
        for layer in range(1, self.final_layer_index + 1):
            self.weights[layer] = self.weights[layer] - (lr * self.Dweights[layer])

    def softmax(self, x):
        """
        Softmax implementation along axis=0

        """
        return np.exp(x) / np.sum(np.exp(x), axis=0)

    @staticmethod
    def _relu(x):
        """
        The relu function that should not be called directly. Vectorized 'NeuralNet.relu' should be called instead, which is defined in 'NeuralNet.__init__' method.

        """
        return np.maximum(0, x)

    @staticmethod
    def _drelu(x):
        """
        The derivative of the relu function that should not be called directly. Vectorized 'NeuralNet.relu' should be called instead, which is defined in 'NeuralNet.__init__' method.

        """
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


if __name__ == '__main__':
    pass