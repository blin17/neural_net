'''
2 Hidden Layer Neural Network
@author: Brian Lin

details:
	input is a circle
	2 inputs + 1 bias term
	2 hidden layer each with 2 nodes
	1 output node
	activation function = hinge loss Max(0,1-s+sc)
'''

import numpy as np

def readdata(file):
	return np.loadtxt(file)

def score(U, f, W, x, b):
	z = np.dot(W,x) + b
	a = f(z)
	return np.dot(U, a)

data = readdata('training_data.tsv')
x = data[:,:2]
y = data[:,2]
W = np.random.rand(*x.shape)

