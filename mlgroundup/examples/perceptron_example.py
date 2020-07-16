# Created by Tristan Bester.
import sys
sys.path.append('../')
from deep_learning import Perceptron
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Create dataset and perceptron model.
X,y = make_blobs(n_samples=50, n_features=2, centers=2, cluster_std=1, random_state=42)
p = Perceptron(2,1,0.01)

# Train perceptron.
p.fit(X, y, 100)

train_set = X.copy() 
labels = y.copy()

# Calculate limits of each dimension of the training set.
X_one_min = X[:,0].min() 
X_one_max = X[:,0].max()
X_two_min = X[:,1].min()
X_two_max = X[:,1].max()

# Calculate the decision boundary of the model.
X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 10), np.linspace(X_two_min, X_two_max,10))
X = [[x1,x2] for x1,x2 in zip(X1.flatten(),X2.flatten())]
Y = p.forward(np.array(X))
Y = Y.reshape(X1.shape)

fig = plt.gcf()
fig.set_size_inches((10,7))

# Plot the decision boundary of the model.
cp1 = plt.contourf(X1, X2, Y, [-1,0, 1, 2], cmap='winter')
plt.scatter(train_set[:,0], train_set[:,1], c=labels, cmap='winter', edgecolors='black')

plt.xlim(X_one_min, X_one_max) 
plt.ylim(X_two_min, X_two_max)  
plt.xlabel('Feature one (Dimension one)')
plt.ylabel('Feature two (Dimension two)')
plt.title('Perceptron decision boundary:')
plt.show()
