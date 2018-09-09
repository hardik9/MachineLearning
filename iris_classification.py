'''
An artificial neural network classifying different types of Iris flowers
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Activation function of neurons
def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _sigmoid_derivative(x):
    return x * (1 - x)

# Getting Data from csv file using pandas module
data = pd.read_csv('iris.data.csv', header=None, usecols=[0,1,2,3])
label = pd.read_csv('iris.data.csv', header=None, usecols=[4])

np_data = np.array(data)
np_labels = np.array(label)

label_list = np_labels.tolist()

# Representing output data in numerical terms
for x in label_list:
    if x[0] == 'Iris-setosa':
        x[0] = 1
    elif x[0] == 'Iris-versicolor':
        x[0] = 2
    elif x[0] == 'Iris-virginica':
        x[0] = 3

np_labels = np.array(label_list)

# setting X as input and y as target output
X = np_data
y = np_labels

# Normalizing data
X = X / np.amax(X, axis=0)
y = y / 3

# Synapses with random initial weight values between -1 and 1
# 4 neurons for input layer, 3 neurons for the hidden layer and 1 for output layer
synapse1 = 2 * np.random.random((4, 3)) - 1
synapse2 = 2 * np.random.random((3, 1)) - 1

# Training
for j in range(40000):

    #Forward propagation of input through layers
    layer0 = X
    layer1 = _sigmoid(np.dot(layer0, synapse1))
    layer2 = _sigmoid(np.dot(layer1, synapse2))

    if j == 0:
        layer2_ini = layer2

    # Backpropagation
    layer2_error = y - layer2
    layer2_delta = layer2_error * _sigmoid_derivative(layer2)

    layer1_error = layer2_delta.dot(synapse2.T)
    layer1_delta = layer1_error * _sigmoid_derivative(layer1)

    # updating weights
    synapse2 += layer1.T.dot(layer2_delta)
    synapse1 += layer0.T.dot(layer1_delta)

    if(j % 5000) == 0:
        print("Error: " + str(np.mean(np.abs(layer2_error))))

plt.subplot(1, 2, 1)
plt.scatter(y, layer2_ini)
plt.title('Before learning')

plt.subplot(1, 2, 2)
plt.scatter(y, layer2)
plt.title('After learning')
plt.show()