# Created by Tristan Bester.
from mlgroundup.ensemble import StackingClassifier
from mlgroundup.supervised import LinearRegression, DecisionTreeClassifier
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# Create the training set.
X, y = make_blobs(n_samples=100, n_features=2, centers=2, cluster_std=4, random_state=42)

# First layer estimators to be used in stacking model. 
base = [LinearRegression(eta=0.000001, n_iters=1000), DecisionTreeClassifier(max_depth=6)]

stack = StackingClassifier(base)
stack.fit(X,y)

train_set = X.copy()
train_y = y.copy()

# Get feature value limits.
x_one_min = X[:, 0].min()
x_one_max = X[:, 0].max()
x_two_min = X[:, 1].min()
x_two_max = X[:, 1].max()

# Generate contour plot.
one, two = np.meshgrid(np.linspace(x_one_min, x_one_max, 100), np.linspace(x_two_min, x_two_max, 100))
X = [[x1,x2] for x1,x2 in zip(one.flatten(), two.flatten())]
y = lambda x: stack.predict(np.array(x))
y = np.array([y(x) for x in X])
y = y.reshape(one.shape)

fig  = plt.gcf()
fig.set_size_inches((5,5))

cp = plt.contour(one,two, y, [0.5])
plt.scatter(train_set[:, 0], train_set[:, 1], c=train_y, cmap='winter')
plt.xlabel('Feature one (Dimension one):')
plt.ylabel('Feature two (Dimension two):')
plt.title('Decision boundary of StackingClassifier:')