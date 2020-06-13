# credit: https://machinelearningmastery.com/implement-backpropagation-algorithm-scratch-python/

from math import exp
from random import seed
from random import random


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
        if i != len(network) - 1:  # if not output layer i.e hidden layers
            for j in range(len(layer)):
                error = 0.0
                for neuron in network[i + 1]:
                    error += (neuron['weights'][j] * neuron['delta'])  # error = (weight_k * error_j)
                errors.append(error)
        else:
            for j in range(len(layer)):  # calculate error signal for output neurons
                neuron = layer[j]
                errors.append(expected[j] - neuron['output'])  # error = (expected - output)

        # error * transfer_derivative(output)
        for j in range(len(layer)):
            neuron = layer[j]
            neuron['delta'] = errors[j] * transfer_derivative(neuron['output'])


def update_weights(network, row, l_rate):
    for i in range(len(network)):
        inputs = row[:-1]
        print(inputs)
        if i != 0:
            inputs = [neuron['output'] for neuron in network[i - 1]]
        for neuron in network[i]:
            for j in range(len(inputs)):
                neuron['weights'][j] += l_rate * neuron['delta'] * inputs[j]
            neuron['weights'][-1] += l_rate * neuron['delta']


# Train a network for a fixed number of epochs
def train_network(network, train, l_rate, n_epoch, n_outputs):
    for epoch in range(n_epoch):
        sum_error = 0
        for row in train:
            outputs = forward_propagate(network, row)  # values outputted from last layer
            expected = [0 for i in range(n_outputs)]  # list of number of classes [0,0]
            expected[row[-1]] = 1  # one hot encoding - set class index position to one
            sum_error += sum([(expected[i] - outputs[i]) ** 2 for i in range(len(expected))])  # sum squared error
            backward_propagate_error(network, expected)
            update_weights(network, row, l_rate)  # stochastic gradient descent
        print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))


# Make a prediction with a network
def predict(network, row):
    outputs = forward_propagate(network, row)
    return outputs.index(max(outputs))  # print index of highest value in list


# test back-propagation of error
seed(1)
dataset = [[2.7810836, 2.550537003, 0],
           [1.465489372, 2.362125076, 0],
           [3.396561688, 4.400293529, 0],
           [1.38807019, 1.850220317, 0],
           [3.06407232, 3.005305973, 0],
           [7.627531214, 2.759262235, 1],
           [5.332441248, 2.088626775, 1],
           [6.922596716, 1.77106367, 1],
           [8.675418651, -0.242068655, 1],
           [7.673756466, 3.508563011, 1]]

n_inputs = len(dataset[0]) - 1
n_outputs = len(set([row[-1] for row in dataset]))  # (0,1)
network = initialize_network(n_inputs, 2, n_outputs)  # inputs, 2 hidden neurons, outputs
train_network(network, dataset, 0.65, 1000, n_outputs)  # error=0.003 - adjust if more training samples added

print('\nMaking Predictions: ')
for row in dataset:
    prediction = predict(network, row)
    print('Expected=%d, Got=%d' % (row[-1], prediction))

