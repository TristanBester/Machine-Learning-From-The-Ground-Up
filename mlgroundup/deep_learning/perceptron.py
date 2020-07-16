# Created by Tristan Bester.
import numpy as np 



class Perceptron(object):
	''' 
    Perceptron model.
    
    Args:
    	input_size (int): Number of neurons in input layer.
    	output_size (int): Number of neurons in output layer.
    	lr (int): The learning rate to be used by the model.
        
    Attributes:
        input_size (int): Number of neurons in input layer.
    	output_size (int): Number of neurons in output layer.
    	lr (int): The learning rate to be used by the model.
    	weights (numpy.ndarray): The weights used in the model.
    	bias: The output value of the bias neuron.
    ''' 
	def __init__(self, input_size, output_size, lr):
		self.input_size = input_size
		self.output_size = output_size
		self.lr = lr
		self.weights = np.random.randn(output_size,input_size)
		self.bias = np.random.uniform()


	def heaviside(self, s):
		'''Heaviside step function.'''
		return (s >= 0).astype(np.int)


	def forward(self, x):
		'''Calculate perceptron output values.'''
		z = self.weights @ x.T
		z += self.bias
		return self.heaviside(z)


	def fit(self, X, Y, n_iters):
		'''Fit the model to the given data.'''
		for i in range(n_iters):
			for x,y in zip(X,Y):
				pred = self.forward(x)
				diff = y - pred

				for row in range(self.weights.shape[0]):
					for col in range(self.weights.shape[1]):
						self.weights[row][col] += self.lr * diff[row] * x[col]
					self.bias += self.lr * (diff[row])
						

