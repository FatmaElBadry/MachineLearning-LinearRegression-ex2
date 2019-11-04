# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 06:01:10 2019

@author: fatma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#=========================================================================
# cost function
def computeCost(X, y, theta):
    z = np.power(((X * theta.T) - y), 2)
#    print('z \n',z)
#    print('m ' ,len(X))
    return np.sum(z) / (2 * len(X))
print('**************************************')

# GD function
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)
    
    for i in range(iters):
        error = (X * theta.T) - y
        
        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))
            
        theta = temp
        cost[i] = computeCost(X, y, theta)
        
    return theta, cost
#=========================================================================

#The path o f the data file
path = 'C:\\Fatma ElBadry\\ML\ML-GitHub-Projects\\MachineLearning-LinearRegression-ex2\\ex2data1.txt'

#Read the data using pandas
data = pd.read_csv(path, header=None, names=['Size', 'Bedrooms', 'Price'])

#show data
print('data = ')
print(data.head(10) )
print()
print('data.describe = ')
print(data.describe())

# rescaling data --> Data Normalization
data = (data - data.mean()) / data.std()

print()
print('data after normalization = ')
print(data.head(10) )
print('data.describe = ')
print(data.describe())

# add ones column --> Used for matrix multiplication to represent the theta0 part as it should be multiplied by ones as theta0 dosen't have x 
data.insert(0, 'Ones', 1)

# separate X (input features data) from y (target variable)
cols = data.shape[1]
X = data.iloc[:,0:cols-1]
y = data.iloc[:,cols-1:cols]

print('**************************************')
print('X data = \n' ,X.head(10) )
print('y data = \n' ,y.head(10) )
print('**************************************')
# convert to matrices and initialize theta
X = np.matrix(X.values)
y = np.matrix(y.values)
theta2 = np.matrix(np.array([0,0,0]))

print('X \n',X)
print('X.shape = ' , X.shape)
print('**************************************')
print('theta2 \n',theta2)
print('theta2.shape = ' , theta2.shape)
print('**************************************')
print('y \n',y)
print('y.shape = ' , y.shape)
print('**************************************')

# initialize variables for learning rate and iterations
alpha = 0.1
iters = 100
# perform linear regression on the data set
g2, cost2 = gradientDescent(X, y, theta2, alpha, iters)
# get the cost (error) of the model
thiscost = computeCost(X, y, g2)

print('g2 = ' , g2)
print('cost2 = ' , cost2[0:50] )
print('computeCost = ' , thiscost)
print('**************************************')
# get best fit line for Size vs. Price
x = np.linspace(data.Size.min(), data.Size.max(), 100)
print('x \n',x)

print('g \n',g2)
f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)
# draw the line for Size vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Size, data.Price, label='Training Data')
ax.legend(loc=2)
ax.set_xlabel('Size')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# get best fit line for Bedrooms vs. Price
x = np.linspace(data.Bedrooms.min(), data.Bedrooms.max(), 100)
print('x \n',x)
print('g \n',g2)
f = g2[0, 0] + (g2[0, 1] * x)
print('f \n',f)
# draw the line for Bedrooms vs. Price
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(x, f, 'r', label='Prediction')
ax.scatter(data.Bedrooms, data.Price, label='Traning Data')
ax.legend(loc=2)
ax.set_xlabel('Bedrooms')
ax.set_ylabel('Price')
ax.set_title('Size vs. Price')

# draw error graph
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(np.arange(iters), cost2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Error vs. Training Epoch')