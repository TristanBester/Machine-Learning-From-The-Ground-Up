# Created by Tristan Bester.
import sys
sys.path.append('../')
from deep_learning import MLP
import matplotlib.pyplot as plt
import numpy as np

# Create dataset and multilayer perceptron model.
training_data = [(np.array([[0],[0]]),0),
				 (np.array([[0],[1]]),1),
		    	 (np.array([[1],[0]]),1),
		    	 (np.array([[1],[1]]),0)]
mlp = MLP([2,2,1])

# Train MLP.
mlp.SGD(training_data, 100000, 1, 0.01)

train_set = np.array([[[0],[0]],
					  [[0],[1]],
					  [[1],[0]],
					  [[1],[1]]])
labels = [[0],[1],[1],[0]]

# Set axis limits.
X_one_min = -0.3
X_one_max = 1.3
X_two_min = -0.3
X_two_max = 1.3

# Calculate the decision boundary of the model.
X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 500), np.linspace(X_two_min, X_two_max,500))
X = [np.array([x1,x2]) for x1,x2 in zip(X1.flatten(),X2.flatten())]
X = [x.reshape(-1,1) for x in X]

outputs = np.array([mlp.feedforward(x) for x in X])
Y = []
for i in outputs:
	if i > 0.5:
		Y.append(1)
	else:
		Y.append(0)
Y = np.array(Y)
Y = Y.reshape(X1.shape)

fig = plt.gcf()
fig.set_size_inches((10,7))

# Plot the decision boundary of the model.
cp1 = plt.contourf(X1, X2, Y, [-1,0, 1, 2], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')

plt.xlim(X_one_min, X_one_max) 
plt.ylim(X_two_min, X_two_max)  
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('MLP decision boundary:')
plt.show()