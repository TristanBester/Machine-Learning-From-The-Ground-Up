# Created by Tristan Bester.
import decision_tree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# Create the training set.
data = make_blobs(n_samples=100, n_features=2, centers=3, cluster_std=5, 
                  random_state=0)
X = data[0]
y = data[1]   

# Instantiate an object of the DecisionTreeClassifier class.
tree = decision_tree.DecisionTreeClassifier()

# Fit the model to the training set.
tree.fit(X,y)

train_set = X.copy() 
labels = y.copy()

# Calculate limits of each dimension of the training set.
X_one_min = X[:,0].min() 
X_one_max = X[:,0].max()
X_two_min = X[:,1].min()
X_two_max = X[:,1].max()

# Calculate the decision boundary of the model.
X1, X2 = np.meshgrid(np.linspace(X_one_min, X_one_max, 500), np.linspace(X_two_min, X_two_max,500))
X = [[x1,x2] for x1,x2 in zip(X1.flatten(),X2.flatten())]
y = lambda x: tree.predict(np.array(x))
Y = []

for x in X:
    Y.append(y(x))

Y = np.array(Y)
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
plt.title(f'Decision boundary of a decision tree with no regularization:')
plt.show()