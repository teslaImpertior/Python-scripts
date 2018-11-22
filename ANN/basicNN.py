# -*- coding: utf-8 -*-
"""
Created on Mon Jul  9 18:44:37 2018

@author: Isaac
"""

### IMPORT LIBRARIES
import numpy as np # mathematical operations
import matplotlib.pyplot as plt # nice graphs
from mpl_toolkits.mplot3d import Axes3D #nice 3d graphs

### GENERATE RANDOM INPUT DATA TO TRAIN ON
observations = 1000

xc=np.random.uniform(low=-10,high=10, size=(observations, 1))
zs=np.random.uniform(-10,10,(observations,1))

inputs = np.column_stack((xc,zs))

print(inputs.shape)


### CREATE THE TARGETS WE WILL AIM AT
noise = np.random.uniform(-1,1,(observations,1))
#targets = 2*xc - 3*zs + 5 + noise
targets = 13*xc + 7*zs - 12 + noise

print(targets.shape)


### PLOT THE TRAINING DATA
targets = targets.reshape(observations,)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(xc, zs, targets)
ax.set_xlabel('xc')
ax.set_ylabel('zs')
ax.set_zlabel('targets')
ax.view_init(azim=100)
plt.show()
targets = targets.reshape(observations,1)

### INITIALISE VARIABLES
init_range = 0.1

weights = np.random.uniform(-init_range,init_range,size=(2,1))
biases = np.random.uniform(-init_range,init_range,size=1)

print(weights)
print(biases)

### SET A LEARNING RATE
learning_rate = 0.001

## MINIMISE THE COST FUNCTION WITH RESPECT TO THE WEIGHTS AND BIASES
### TRAINING OUR MODEL
for i in range (10000):
    outputs = np.dot(inputs,weights) + biases
    deltas = outputs - targets
    loss=np.sum(deltas ** 2) / 2 / observations #adjusting the loss
    
    print(loss)
    #update weights and biases ready for the next loop
    deltas_scaled = deltas / observations
    #following gradient descent logic
    weights = weights - learning_rate * np.dot(inputs.T,deltas_scaled)
    biases = biases - learning_rate * np.sum(deltas_scaled)

#print weights and biases
print(weights, biases)

#plot linear regression line
plt.plot(outputs,targets)
plt.xlabel('outputs')
plt.ylabel('targets')
plt.show()