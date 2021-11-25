from nn import NeuralNetwork
import random
import numpy as np


np.printoptions(suppress=True)
nn = NeuralNetwork(2, 4, 1, learning_rate=0.1)
epochs = 5000
show_progress = False

training_data = [
    {"inputs": [0, 0], "targets": [0]},
    {"inputs": [0, 1], "targets": [1]},
    {"inputs": [1, 0], "targets": [1]},
    {"inputs": [1, 1], "targets": [0]},
]

# Learn something!
for i in range(0, epochs):
    x = np.random.randint(0, len(training_data) - 1)
    curr_data = training_data[x]
    nn.backpropagation(curr_data["inputs"], curr_data["targets"])

    # Print display output error per epoch?
    if show_progress:
        print("Epoch: " + str(i + 1) + " - error rate = " + str(nn.output_error))

# Check how NN performs with inputs after learning
for data in training_data:
    print("-----------------------------")
    print("Testing for :" + str(data["inputs"]))
    print("Output is: " + str(nn.feedforward(data["inputs"])))
    print("Target was: " + str(data["targets"]))
