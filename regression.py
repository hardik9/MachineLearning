'''
An artificial neural network (MLP) learning a function (y = x^2),
using backpropagation algorithm. 
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import neurolab as nl

# generating data points
min = -50
max = 50

# even numbers between -50 and 50
x = []
for i in range(min, max):
    if i % 2 == 0:
        x.append(i)

x = np.array(x, dtype=float)

# y = x^2
y = np.square(x)

# normalizing data
y /= np.linalg.norm(y)

# reshaping data points
input = x.reshape(x.size, 1)
output = y.reshape(y.size, 1)

# creating neural network
nn = nl.net.newff( [[min, max]], [4,2,1] )

# training neural network
nn.trainf = nl.train.train_gd
error = nn.train(input, output, epochs=40000, show=500, goal=0.005)

# generating data for testing the neural network
test_x = []

# odd numbers between -50 and 50
for i in range(min, max):
    if i % 2 != 0:
        test_x.append(i)

test_x = np.array(test_x, dtype=float)

# y = x^2
test_y = np.square(test_x)
test_y /= np.linalg.norm(test_y)

# reshaping
test_input = test_x.reshape(test_x.size, 1)
test_output = test_y.reshape(test_y.size, 1)

# getting test result
result = nn.sim(test_input)
prediction = result.reshape(test_output.size)

# plotting test result
plt.plot( x, y, '.', test_x, prediction,'-')
plt.title('Function curve v/s Prediction curve')
plt.show()
