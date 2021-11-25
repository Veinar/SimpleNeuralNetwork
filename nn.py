###
###     Date: 25/11/2021
###     Author: Konrad (Veinar)
###

from functools import singledispatchmethod
import numpy as np


class NeuralNetwork:

    # Constructor
    def __init__(self, num_Input, num_Hidden, num_Output, learning_rate=0.1) -> None:
        # Get values from args (size/shape of NN)
        self.input_nodes = num_Input
        self.hidden_nodes = num_Hidden
        self.output_nodes = num_Output

        # Randomize weights on layer Input-Hidden
        self.weights_ih = np.random.default_rng(np.random.randint(1, 100)).random(
            (self.hidden_nodes, self.input_nodes)
        )
        # self.weights_ih = np.ones((self.hidden_nodes, self.input_nodes))

        # Randomize weights in layer Hidden-Output
        self.weights_ho = np.random.default_rng(np.random.randint(1, 100)).random(
            (self.output_nodes, self.hidden_nodes)
        )
        # self.weights_ho = np.ones((self.output_nodes, self.hidden_nodes))

        # Set BIAS for layers Hidden and Output
        self.bias_h = np.ones((self.hidden_nodes, 1))
        # self.bias_h = np.random.default_rng(np.random.randint(1, 100)).random(
        #    (self.hidden_nodes, 1)
        # )
        self.bias_o = np.ones((self.output_nodes, 1))
        # self.bias_o = np.random.default_rng(np.random.randint(1, 100)).random(
        #    (self.output_nodes, 1)
        # )
        self.bias_h *= -1
        self.bias_o *= -1

        # Declare learning rate
        self.learning_rate = learning_rate

        # Set variables for errors per every layer
        self.hidden_error = None
        self.output_error = None

        # Set variables for layers after sigmoid function
        self.output = None
        self.hidden = None

    # Put data into NN
    def feedforward(self, input):
        # Make vertical array out of input
        input = np.array(input)
        input = np.vstack(input)

        self.hidden = np.dot(self.weights_ih, input)
        self.hidden = np.add(self.hidden, self.bias_h)
        # Activation function for hidden layer
        self.hidden = self.sigmoid(self.hidden)

        self.output = np.dot(self.weights_ho, self.hidden)
        self.output = np.add(self.output, self.bias_o)
        # Activation function for output layer
        self.output = self.sigmoid(self.output)

        return self.output

    # Activation function
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    # Devirative for activation function
    def derivative_sigmoid(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    # Simplified diverative for activation function (for use in backpropagation)
    def calculate_gradient(self, x):
        return x * (1 - x)

    # Backpropagation of NN
    def backpropagation(self, inputs, targets) -> None:
        # Feed NN
        self.output = self.feedforward(inputs)

        # TODO: delete this
        np.printoptions(suppress=True)

        # Make vertical matrix out of input
        input = np.array(inputs)
        input = np.vstack(input)

        # Make vertical matrix out of targets
        target = np.array(targets)
        target = np.vstack(target)

        # Calculate output error which is diffrence between target and output
        # ERROR = TARGET - OUTPUT
        self.output_error = np.subtract(target, self.output)
        # OK! [rows = output_num, cols = 1]

        # Calculate hidden layer errors
        transposed_weights_ho = np.transpose(self.weights_ho)
        self.hidden_error = np.dot(transposed_weights_ho, self.output_error)
        # OK! [rows = hidden_num, cols = 1]

        # -----------------------------------------------------------------
        # Calculate delta to weights in HO layer
        # -----------------------------------------------------------------

        # DeltaHO = LEARN_RATE * output_error * (output * (1 - output)) -dot- hidden^T
        delta_weights_ho = np.multiply(self.output_error, self.learning_rate)
        delta_bias_o = self.calculate_gradient(delta_weights_ho)
        delta_weights_ho = self.calculate_gradient(delta_weights_ho)
        hidden_transposed = np.transpose(self.hidden)
        delta_weights_ho = np.dot(delta_weights_ho, hidden_transposed)
        # OK! same size as weights_ho

        # -----------------------------------------------------------------
        # Calculate delta to weights in IH layer
        # -----------------------------------------------------------------

        # DeltaIH = LEARN_RATE * hidden_error * (hidden * (1 - hidden)) -dot- Input^T
        delta_weights_ih = np.multiply(self.hidden_error, self.learning_rate)
        delta_bias_h = self.calculate_gradient(delta_weights_ih)
        delta_weights_ih = self.calculate_gradient(delta_weights_ih)
        input_transposed = np.transpose(input)
        delta_weights_ih = np.dot(delta_weights_ih, input_transposed)
        # OK! same size as weights_ih

        # Adjust weights of HO layer
        self.weights_ho = np.add(self.weights_ho, delta_weights_ho)

        # Adjust weights of IH layer
        self.weights_ih = np.add(self.weights_ih, delta_weights_ih)

        # Adjust BIAS for Hidden layer
        self.bias_h = np.add(self.bias_h, delta_bias_h)

        # Adjust BIAS for Output layer
        self.bias_o = np.add(self.bias_o, delta_bias_o)
