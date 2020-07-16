# Created by Tristan Bester.
import random
import numpy as np 



class MLP(object):
	'''
	This implementation is based on the neural network implementation in chapter 2
	of the book "Neural Networks and Deep Learning" that is available 
	here: http://neuralnetworksanddeeplearning.com/chap2.html

	This implementation has been altered to suit the article
	available here: 

 	Args:
    	sizes (list): A list specifying the number of neurons in each layer.
        
    Attributes:
        sizes (list): A list specifying the number of neurons in each layer.
    	weights (list): A list storing the models weights.
    	biases (list): A list storing the models biases.
    	num_layers (int): The number of layers in the network.
	'''
	def __init(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(i, 1) for i in sizes[1:]]
		self.weights = [np.random.randn(nxt, lst) 
						for lst, nxt in zip(sizes[:-1], sizes[1:])]


	def sigmoid(self, z):
		'''The Sigmoid function.'''
		return 1.0/(1.0+np.exp(-z))


	def sigmoid_prime(self, z):
		'''The derivative of the Sigmoid function.'''
		return self.sigmoid(z)*(1-self.sigmoid(z))


	def feedforward(self, a):
		'''Calculate activations of the output neurons.'''
		for i in range(len(self.weights)):
			a = np.dot(self.weights[i], a) + self.biases[i]
			a = self.sigmoid(a)
		return a


	def SGD(self, training_data, epochs, mini_batch_size, eta):
		'''Train model using Stochastic gradient descent.'''
		for j in range(epochs):
			for x,y in training_data:
				delta_nabla_b, delta_nabla_w = self.backprop(x, y)

				self.weights = [w - (eta*dw).reshape(w.shape) for w, dw in zip(self.weights, delta_nabla_w)]	
				self.biases = [b - eta*nb for b, nb in zip(self.biases, delta_nabla_b)]
		

	def backprop(self, x, y):
		'''Backpropagation.'''
		nabla_w = [0] * len(self.weights)
		nabla_b = [0] * len(self.biases)

		a = x
		activations = [x]
		sums = [None]
		for i in range(len(self.weights)):
			a = np.dot(self.weights[i], a) + self.biases[i]
			sums.append(a)
			a = self.sigmoid(a)
			activations.append(a)

		delta = (activations[-1] - y) * self.sigmoid_prime(sums[-1])
		nabla_w[-1] = np.multiply(activations[-2], delta)
		nabla_b[-1] = delta

		for i in range(len(self.sizes)-2, 0, -1):
			delta = np.dot(self.weights[i].T, delta)
			delta = np.multiply(delta, self.sigmoid_prime(sums[i]))
			nabla_b[i-1] = delta
			nabla_w[i-1] = np.dot(delta, activations[i-1].T)

		return (nabla_b, nabla_w)