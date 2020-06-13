# credit: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from random import random
from math import exp


# Initialize a network
def initialize_network(n_inputs, n_hidden, n_outputs):
    network = list()
    hidden_layer = [{'weights': [random() for i in range(n_inputs + 1)]} for i in range(n_hidden)]
    network.append(hidden_layer)
    output_layer = [{'weights': [random() for i in range(n_hidden + 1)]} for i in range(n_outputs)]
    network.append(output_layer)
    return network


# Calculate neuron activation for an input
def activate(weights, inputs):
    activation = weights[-1]
    for i in range(len(weights) - 1):
        activation += weights[i] * inputs[i]
    return activation


# Transfer neuron activation
def transfer(activation):
    return 1.0 / (1.0 + exp(-activation))


# Calculate the derivative of an neuron output using sigmoid function
def transfer_derivative(output):
    return output * (1.0 - output)


# Forward propagate input to a network output
def forward_propagate(network, row):
    inputs = row
    for layer in network:
        new_inputs = []
        for neuron in layer:
            activation = activate(neuron['weights'], inputs)
            neuron['output'] = transfer(activation)
            new_inputs.append(neuron['output'])
        inputs = new_inputs
    return inputs


# Backpropagate error and store in neurons
def backward_propagate_error(network, expected):
    for i in reversed(range(len(network))):
        layer = network[i]  # starts at output layer
        errors = list()
        if i != len(network) - 1:  # if not hidden ;ayer
            for j in range(len(layer)):
                print(len(layer))
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])
                errors.append(error)
        else:
            for j in range(len(layer)):
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


# test forward propagation
network = [[{'weights': [0.13436424411240122, 0.8474337369372327, 0.763774618976614]}], # 2 inputs to hidden neuron with bias
           [{'weights': [0.2550690257394217, 0.49543508709194095]},  # hidden to output neuron 1 with weight and bias
            {'weights': [0.4494910647887381, 0.651592972722763]}]]  # hidden to output neuron 2 with weight and bias

row = [1, 0, None]  # inputs 1, 0 and no weight for input
output = forward_propagate(network, row)
print(output)

expected = [0, 1]
backward_propagate_error(network, expected)
for layer in network:
	print(layer)